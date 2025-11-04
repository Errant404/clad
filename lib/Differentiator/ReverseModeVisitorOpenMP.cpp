#include "clad/Differentiator/MultiplexExternalRMVSource.h"
#include "clad/Differentiator/ReverseModeVisitor.h"
#include <clang/AST/OpenMPClause.h>
#include <clang/AST/StmtOpenMP.h>
#include <clang/Basic/OpenMPKinds.h>
#include <llvm/Frontend/OpenMP/OMP.h.inc>
#include <llvm/Support/ErrorHandling.h>

#include <numeric>
#include <omp.h>

using namespace clang;
using namespace llvm::omp;

namespace clad {
/* Static OpenMP scheduler, identical to what LLVM would use. Each thread gets
   one chunk of consecutive iterations. The number of iterations per chunk is
   aproximately trip_count/num_threads. If the trip count can not be evenly
   divided among threads, the first few threads get one extra iteration.
   As long as the number of threads stays constant, and when called by the
   same thread, this subroutine will always return the same threadstart and
   threadend when given the same imin,imax,istride as input. */
static void GetStaticSchedule(int lo, int hi, int stride, int* threadlo,
                              int* threadhi) {
  int trip_count = ((hi - lo + stride) / stride);
  trip_count = std::max(trip_count, 0);

  int nth = omp_get_num_threads();
  int tid = omp_get_thread_num();

  if (trip_count < nth) {
    /* fewer iterations than threads. some threads will get one iteration,
       the other threads will get nothing. */
    if (tid < trip_count) {
      /* do one iteration */
      *threadlo = lo + tid * stride;
      *threadhi = *threadlo;
    } else {
      /* do nothing */
      *threadhi = 0;
      *threadlo = *threadhi + stride;
    }
  }
  /* at least one iteration per thread. since the total number of iterations may
     not be evenly dividable by the number of threads, there will be a few extra
     iterations. the first few threads will each get one of those, which results
     in some offsetts that are applied to the start and end of the chunks. */
  else {
    int chunksize = trip_count / nth;
    int extras = trip_count % nth;
    int tidextras;
    int incr;
    if (tid < extras) {
      tidextras = tid;
      incr = 0;
    } else {
      tidextras = extras;
      incr = stride;
    }
    *threadlo = lo + (tid * chunksize + tidextras) * stride;
    *threadhi = *threadlo + chunksize * stride - incr;
  }
}

ForStmt* ReverseModeVisitor::DifferentiateCanonicalLoop(ForStmt* S) {
  if (!S)
    return nullptr;

  ASTContext& Ctx = m_Sema.getASTContext();
  SourceLocation Loc = S->getForLoc();

  // Extract loop components
  Stmt* Init = S->getInit();
  Expr* Cond = S->getCond();
  Expr* Inc = S->getInc();
  Stmt* Body = S->getBody();

  if (!Init || !Cond || !Inc || !Body)
    return nullptr;

  // Extract loop variable from initialization
  // In canonical form: int i = istart or existing var assigned
  VarDecl* LoopVar = nullptr;
  Expr* InitValue = nullptr;

  if (auto* DS = dyn_cast<DeclStmt>(Init)) {
    if (DS->isSingleDecl()) {
      LoopVar = dyn_cast<VarDecl>(DS->getSingleDecl());
      if (LoopVar && LoopVar->hasInit())
        InitValue = LoopVar->getInit();
    }
  } else if (auto* BinOp = dyn_cast<BinaryOperator>(Init)) {
    if (BinOp->getOpcode() == BO_Assign) {
      if (auto* DRE = dyn_cast<DeclRefExpr>(BinOp->getLHS())) {
        LoopVar = dyn_cast<VarDecl>(DRE->getDecl());
        InitValue = BinOp->getRHS();
      }
    }
  }

  if (!LoopVar || !InitValue)
    return nullptr;

  // Extract condition: i < iend, i <= iend, i > istart, i >= istart
  auto* CondBinOp = dyn_cast<BinaryOperator>(Cond);
  if (!CondBinOp)
    return nullptr;

  BinaryOperatorKind CondOp = CondBinOp->getOpcode();
  Expr* CondLHS = CondBinOp->getLHS();
  Expr* CondRHS = CondBinOp->getRHS();

  // Verify LHS is loop variable
  auto* CondVarRef = dyn_cast<DeclRefExpr>(CondLHS->IgnoreParenImpCasts());
  if (!CondVarRef || CondVarRef->getDecl() != LoopVar)
    return nullptr;

  Expr* BoundExpr = CondRHS;

  // Determine loop direction and extract step
  bool IsIncreasing = false;
  Expr* StepExpr = nullptr;
  bool IsStepNegated = false;

  // Analyze increment expression
  if (auto* UnaryInc = dyn_cast<UnaryOperator>(Inc)) {
    UnaryOperatorKind UOp = UnaryInc->getOpcode();
    if (UOp == UO_PreInc || UOp == UO_PostInc) {
      IsIncreasing = true;
      StepExpr = IntegerLiteral::Create(
          Ctx, llvm::APInt(Ctx.getIntWidth(LoopVar->getType()), 1),
          LoopVar->getType(), Loc);
    } else if (UOp == UO_PreDec || UOp == UO_PostDec) {
      IsIncreasing = false;
      StepExpr = IntegerLiteral::Create(
          Ctx, llvm::APInt(Ctx.getIntWidth(LoopVar->getType()), 1),
          LoopVar->getType(), Loc);
    } else {
      return nullptr;
    }
  } else if (auto* BinInc = dyn_cast<BinaryOperator>(Inc)) {
    BinaryOperatorKind IncOp = BinInc->getOpcode();
    if (IncOp == BO_AddAssign) {
      IsIncreasing = true;
      StepExpr = BinInc->getRHS();
    } else if (IncOp == BO_SubAssign) {
      IsIncreasing = false;
      StepExpr = BinInc->getRHS();
    } else {
      return nullptr;
    }
  } else if (auto* CallInc = dyn_cast<CXXOperatorCallExpr>(Inc)) {
    OverloadedOperatorKind OOK = CallInc->getOperator();
    if (OOK == OO_PlusPlus) {
      IsIncreasing = true;
      StepExpr = IntegerLiteral::Create(
          Ctx, llvm::APInt(Ctx.getIntWidth(LoopVar->getType()), 1),
          LoopVar->getType(), Loc);
    } else if (OOK == OO_MinusMinus) {
      IsIncreasing = false;
      StepExpr = IntegerLiteral::Create(
          Ctx, llvm::APInt(Ctx.getIntWidth(LoopVar->getType()), 1),
          LoopVar->getType(), Loc);
    } else if (OOK == OO_PlusEqual) {
      IsIncreasing = true;
      StepExpr = CallInc->getArg(1);
    } else if (OOK == OO_MinusEqual) {
      IsIncreasing = false;
      StepExpr = CallInc->getArg(1);
    } else {
      return nullptr;
    }
  } else {
    return nullptr;
  }

  if (!StepExpr)
    return nullptr;

  // Verify condition matches loop direction
  bool CondMatchesDirection = false;
  if (IsIncreasing && (CondOp == BO_LT || CondOp == BO_LE))
    CondMatchesDirection = true;
  else if (!IsIncreasing && (CondOp == BO_GT || CondOp == BO_GE))
    CondMatchesDirection = true;

  if (!CondMatchesDirection)
    return nullptr;

  // Now construct the reversed loop
  // For increasing loop: for (i = istart; i < iend; i += step)
  //   becomes: for (i = iend - 1; i >= istart; i -= step)
  // For decreasing loop: for (i = istart; i > iend; i -= step)
  //   becomes: for (i = iend + 1; i <= istart; i += step)

  QualType LoopVarType = LoopVar->getType();

  // Calculate new initial value
  Expr* NewInitValue = nullptr;
  if (IsIncreasing) {
    // iend - 1 (or iend if original was <=)
    Expr* OneExpr = IntegerLiteral::Create(
        Ctx, llvm::APInt(Ctx.getIntWidth(LoopVarType), 1), LoopVarType, Loc);

    if (CondOp == BO_LT) {
      // i < iend becomes i >= iend - 1
      NewInitValue = BuildOp(BO_Sub, BoundExpr, OneExpr);
    } else {
      // i <= iend becomes i >= iend  (no adjustment needed)
      NewInitValue = BoundExpr;
    }
  } else {
    // iend + 1 (or iend if original was >=)
    Expr* OneExpr = IntegerLiteral::Create(
        Ctx, llvm::APInt(Ctx.getIntWidth(LoopVarType), 1), LoopVarType, Loc);

    if (CondOp == BO_GT) {
      // i > iend becomes i <= iend + 1
      NewInitValue = BuildOp(BO_Add, BoundExpr, OneExpr);
    } else {
      // i >= iend becomes i <= iend (no adjustment needed)
      NewInitValue = BoundExpr;
    }
  }

  if (!NewInitValue)
    return nullptr;

  // Create new initialization
  Stmt* NewInit = nullptr;
  if (isa<DeclStmt>(Init)) {
    // Create a new VarDecl with the new initial value
    VarDecl* NewLoopVar =
        BuildVarDecl(LoopVar->getType(), LoopVar->getIdentifier());
    NewLoopVar->setInit(NewInitValue);
    NewLoopVar->setInitStyle(LoopVar->getInitStyle());

    NewInit = BuildDeclStmt(NewLoopVar);
  } else {
    // Assignment form
    DeclRefExpr* VarRef = BuildDeclRef(LoopVar);

    NewInit = BuildOp(BO_Assign, VarRef, NewInitValue);
  }

  if (!NewInit)
    return nullptr;

  // Create new condition
  DeclRefExpr* NewCondVarRef = BuildDeclRef(LoopVar);

  BinaryOperatorKind NewCondOp;
  if (IsIncreasing)
    NewCondOp = (CondOp == BO_LT || CondOp == BO_LE) ? BO_GE : BO_LE;
  else
    NewCondOp = (CondOp == BO_GT || CondOp == BO_GE) ? BO_LE : BO_GE;

  Expr* NewCond = BuildOp(NewCondOp, NewCondVarRef, InitValue);
  if (!NewCond)
    return nullptr;

  // Create new increment
  DeclRefExpr* NewIncVarRef = BuildDeclRef(LoopVar);

  Expr* NewInc = nullptr;
  if (auto* UnaryInc = dyn_cast<UnaryOperator>(Inc)) {
    UnaryOperatorKind UOp = UnaryInc->getOpcode();
    UnaryOperatorKind NewUOp;
    if (UOp == UO_PreInc)
      NewUOp = UO_PreDec;
    else if (UOp == UO_PostInc)
      NewUOp = UO_PostDec;
    else if (UOp == UO_PreDec)
      NewUOp = UO_PreInc;
    else if (UOp == UO_PostDec)
      NewUOp = UO_PostInc;
    else
      return nullptr;

    NewInc = BuildOp(NewUOp, NewIncVarRef);
  } else if (isa<BinaryOperator>(Inc) || isa<CXXOperatorCallExpr>(Inc)) {
    // For compound assignment, reverse the operation
    BinaryOperatorKind NewIncOp = IsIncreasing ? BO_SubAssign : BO_AddAssign;
    NewInc = BuildOp(NewIncOp, NewIncVarRef, StepExpr);
  }

  if (!NewInc)
    return nullptr;

  // Create the new ForStmt
  auto NewCondResult = Sema::ConditionResult(
      m_Sema.ActOnCondition(nullptr, Loc, NewCond, Sema::ConditionKind::Boolean,
                            /*MissingOK=*/true));
  Sema::FullExprArg FullNewInc = m_Sema.MakeFullDiscardedValueExpr(NewInc);

  StmtResult Result = m_Sema.ActOnForStmt(S->getForLoc(), S->getLParenLoc(),
                                          NewInit, NewCondResult, FullNewInc,
                                          S->getRParenLoc(), Clone(Body));

  if (Result.isInvalid())
    return nullptr;

  return dyn_cast<ForStmt>(Result.get());
}

std::pair<OMPClause*, OMPClause*>
ReverseModeVisitor::VisitOMPReductionClause(const OMPReductionClause* C) {
  llvm::SmallVector<Expr*, 16> Vars;
  llvm::SmallVector<Expr*, 16> DiffVars;
  Vars.reserve(C->varlist_size());
  DiffVars.reserve(C->varlist_size());
  for (const auto* Var : CLAD_COMPAT_CLANG20_getvarlist(C)) {
    DiffVars.push_back(Visit(Var).getExpr_dx());
    Vars.push_back(Clone(Var));
  }
  CXXScopeSpec ReductionIdScopeSpec;
  ReductionIdScopeSpec.Adopt(C->getQualifierLoc());
  DeclarationNameInfo NameInfo = C->getNameInfo();
  return {CLAD_COMPAT_CLANG19_SemaOpenMP(m_Sema).ActOnOpenMPReductionClause(
              Vars, CLAD_COMPAT_CLANG21_getModifier(C), C->getBeginLoc(),
              C->getLParenLoc(), C->getModifierLoc(), C->getColonLoc(),
              C->getEndLoc(), ReductionIdScopeSpec, NameInfo),
          CLAD_COMPAT_CLANG19_SemaOpenMP(m_Sema).ActOnOpenMPFirstprivateClause(
              DiffVars, C->getBeginLoc(), C->getLParenLoc(), C->getEndLoc())};
}
StmtDiff ReverseModeVisitor::VisitOMPExecutableDirective(
    const OMPExecutableDirective* D) {
  llvm::SmallVector<OMPClause*, 16> OrigClauses;
  llvm::SmallVector<OMPClause*, 16> DiffClauses;
  ArrayRef<OMPClause*> Clauses = D->clauses();
  OrigClauses.reserve(Clauses.size());
  DiffClauses.reserve(Clauses.size());
  for (auto* I : Clauses) {
    assert(I);
    CLAD_COMPAT_CLANG19_SemaOpenMP(m_Sema).StartOpenMPClause(
        I->getClauseKind());
    auto ClausePair = Visit(I);
    CLAD_COMPAT_CLANG19_SemaOpenMP(m_Sema).EndOpenMPClause();
    OrigClauses.push_back(ClausePair.first);
    // DiffClauses.push_back(ClausePair.first);
    assert(ClausePair.second);
    DiffClauses.push_back(ClausePair.second);
  }
  // We only support static schedule for now, so force to use it.
  OrigClauses.push_back(
      CLAD_COMPAT_CLANG19_SemaOpenMP(m_Sema).ActOnOpenMPScheduleClause(
          OMPC_SCHEDULE_MODIFIER_unknown, OMPC_SCHEDULE_MODIFIER_unknown,
          OMPC_SCHEDULE_static, nullptr, noLoc, noLoc, noLoc, noLoc, noLoc,
          noLoc, noLoc));
  DiffClauses.push_back(
      CLAD_COMPAT_CLANG19_SemaOpenMP(m_Sema).ActOnOpenMPScheduleClause(
          OMPC_SCHEDULE_MODIFIER_unknown, OMPC_SCHEDULE_MODIFIER_unknown,
          OMPC_SCHEDULE_static, nullptr, noLoc, noLoc, noLoc, noLoc, noLoc,
          noLoc, noLoc));
  StmtDiff AssociatedSDiff;
  if (D->hasAssociatedStmt() && D->getAssociatedStmt()) {
    auto processBody = [&]() -> StmtDiff {
      CLAD_COMPAT_CLANG19_SemaOpenMP(m_Sema).ActOnOpenMPRegionStart(
          D->getDirectiveKind(), getCurrentScope());
      StmtDiff Body;
      {
        Sema::CompoundScopeRAII CompoundScope(m_Sema);
        const auto* CS = D->getInnermostCapturedStmt()->getCapturedStmt();
        if (isOpenMPLoopDirective(D->getDirectiveKind())) {
          const auto* FS = cast<ForStmt>(CS);
          const auto* init = FS->getInit();
          StmtDiff initResult =
              init ? DifferentiateSingleStmt(init) : StmtDiff{};

          StmtDiff condVarRes;
          VarDecl* condVarClone = nullptr;
          if (FS->getConditionVariable()) {
            condVarRes =
                DifferentiateSingleStmt(FS->getConditionVariableDeclStmt());
            if (isa<DeclStmt>(condVarRes.getStmt())) {
              Decl* decl =
                  cast<DeclStmt>(condVarRes.getStmt())->getSingleDecl();
              condVarClone = cast<VarDecl>(decl);
            }
          }
          StmtDiff condDiff;
          StmtDiff condExprDiff;
          if (FS->getCond())
            std::tie(condDiff, condExprDiff) =
                DifferentiateSingleExpr(FS->getCond());

          StmtDiff incDiff;
          StmtDiff incExprDiff;
          if (const Expr* inc = FS->getInc()) {
            std::tie(incDiff, incExprDiff) = DifferentiateSingleExpr(inc);
            auto CommaJoin = [this](Expr* Acc, Stmt* S) {
              Expr* E = cast<Expr>(S);
              return BuildOp(BO_Comma, E, BuildParens(Acc));
            };
            auto* Additional = cast<CompoundStmt>(incDiff.getStmt());
            incDiff.updateStmt(std::accumulate(
                Additional->body_rbegin(), Additional->body_rend(),
                incExprDiff.getExpr(), CommaJoin));
          }
          auto BodyDiff = Visit(FS->getBody());
          Stmt* Forward = new (m_Context)
              ForStmt(m_Context, initResult.getStmt(), condExprDiff.getExpr(),
                      condVarClone, incDiff.getExpr(), BodyDiff.getStmt(),
                      noLoc, noLoc, noLoc);

          Stmt* Reverse = new (m_Context)
              ForStmt(m_Context, initResult.getStmt(), condExprDiff.getExpr(),
                      condVarClone, incDiff.getExpr(), BodyDiff.getStmt_dx(),
                      noLoc, noLoc, noLoc);
          return {Forward, Reverse};
        }
        return Visit(CS);
      }
    };
    Stmt* Forward =
        CLAD_COMPAT_CLANG19_SemaOpenMP(m_Sema)
            .ActOnOpenMPRegionEnd(processBody().getStmt(), OrigClauses)
            .get();
    Stmt* Reverse =
        CLAD_COMPAT_CLANG19_SemaOpenMP(m_Sema)
            .ActOnOpenMPRegionEnd(processBody().getStmt_dx(), DiffClauses)
            .get();
    AssociatedSDiff = {Forward, Reverse};
  }
  DeclarationNameInfo DirName;
  OpenMPDirectiveKind CancelRegion = OMPD_unknown;
  return {CLAD_COMPAT_CLANG19_SemaOpenMP(m_Sema)
              .ActOnOpenMPExecutableDirective(
                  D->getDirectiveKind(), DirName, CancelRegion, OrigClauses,
                  AssociatedSDiff.getStmt(), D->getBeginLoc(), D->getEndLoc())
              .get(),
          CLAD_COMPAT_CLANG19_SemaOpenMP(m_Sema)
              .ActOnOpenMPExecutableDirective(D->getDirectiveKind(), DirName,
                                              CancelRegion, DiffClauses,
                                              AssociatedSDiff.getStmt_dx(),
                                              D->getBeginLoc(), D->getEndLoc())
              .get()};
}
StmtDiff ReverseModeVisitor::VisitOMPParallelForDirective(
    const clang::OMPParallelForDirective* D) {
  DeclarationNameInfo DirName;
  CLAD_COMPAT_CLANG19_SemaOpenMP(m_Sema).StartOpenMPDSABlock(
      OMPD_parallel_for, DirName, getCurrentScope(), D->getBeginLoc());
  StmtDiff SDiff = VisitOMPExecutableDirective(D);
  CLAD_COMPAT_CLANG19_SemaOpenMP(m_Sema).EndOpenMPDSABlock(SDiff.getStmt());
  return SDiff;
}
} // namespace clad