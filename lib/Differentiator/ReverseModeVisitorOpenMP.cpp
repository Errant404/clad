#include "ConstantFolder.h"
#include "clad/Differentiator/MultiplexExternalRMVSource.h"
#include "clad/Differentiator/ReverseModeVisitor.h"

#include <clang/AST/OpenMPClause.h>
#include <clang/AST/StmtOpenMP.h>
#include <clang/Basic/OpenMPKinds.h>
#include <llvm/Frontend/OpenMP/OMP.h.inc>
#include <llvm/Support/ErrorHandling.h>

using namespace clang;
using namespace llvm::omp;

namespace clad {
StmtDiff ReverseModeVisitor::DifferentiateCanonicalLoop(const ForStmt* S) {
  // OpenMP canonical loops have the form:
  // for (init-expr; test-expr; incr-expr) structured-block
  // where init-expr: var = lb
  //       test-expr: var relational-op ub
  //       incr-expr: ++var, var++, --var, var--, var += incr, var -= incr,
  //                  var = var + incr, var = incr + var, var = var - incr

  // Extract loop components
  const Stmt* Init = S->getInit();
  const Expr* Cond = S->getCond();
  const Expr* Inc = S->getInc();
  const Stmt* Body = S->getBody();

  assert(Init && Cond && Inc && Body);

  // Extract loop variable, lower bound, upper bound, and stride
  const VarDecl* LoopVarDecl = nullptr;
  Expr* LowerBound = nullptr;
  Expr* UpperBound = nullptr;
  Expr* Stride = nullptr;

  // Parse init expression: var = lb
  // Handle both DeclStmt (int i = 0) and BinaryOperator (i = 0)
  if (const auto* DS = dyn_cast<DeclStmt>(Init)) {
    if (DS->isSingleDecl()) {
      LoopVarDecl = dyn_cast<VarDecl>(DS->getSingleDecl());
      if (LoopVarDecl && LoopVarDecl->hasInit())
        LowerBound = Clone(LoopVarDecl->getInit());
    }
  } else if (const auto* BO = dyn_cast<BinaryOperator>(Init)) {
    if (BO->getOpcode() == BO_Assign) {
      const Expr* LHS = BO->getLHS()->IgnoreImplicitAsWritten();
      if (const auto* DRE = dyn_cast<DeclRefExpr>(LHS)) {
        LoopVarDecl = dyn_cast<VarDecl>(DRE->getDecl());
        LowerBound = Clone(BO->getRHS());
      }
    }
  }

  // Parse condition expression: var < ub, var <= ub, etc.
  // Need to handle implicit casts and other wrappers
  const Expr* CondExpr = Cond->IgnoreImplicitAsWritten();
  if (const auto* BO = dyn_cast<BinaryOperator>(CondExpr)) {
    // Try to match: var relop ub
    const Expr* LHS = BO->getLHS()->IgnoreImplicitAsWritten();
    const Expr* RHS = BO->getRHS()->IgnoreImplicitAsWritten();

    if (const auto* DRE = dyn_cast<DeclRefExpr>(LHS)) {
      if (DRE->getDecl() == LoopVarDecl) {
        // Pattern: var < ub, var <= ub, var != ub
        UpperBound = Clone(BO->getRHS());
      }
    } else if (const auto* DRE = dyn_cast<DeclRefExpr>(RHS)) {
      if (DRE->getDecl() == LoopVarDecl) {
        // Pattern: ub > var, ub >= var, ub != var (reversed)
        UpperBound = Clone(BO->getLHS());
      }
    }
  }

  // Parse increment expression and extract stride
  // Supported forms: ++var, var++, var += incr, var = var + incr, etc.
  Stride = ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context,
                                             /*val=*/1); // Default stride
  bool IsIncrement = true; // true for +=, false for -=

  if (const auto* UO = dyn_cast<UnaryOperator>(Inc)) {
    // ++var or var++, --var or var--
    IsIncrement =
        (UO->getOpcode() == UO_PreInc || UO->getOpcode() == UO_PostInc);
    Stride = ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context,
                                               /*val=*/1);
  } else if (const auto* BO = dyn_cast<BinaryOperator>(Inc)) {
    if (BO->getOpcode() == BO_AddAssign) {
      IsIncrement = true;
      Stride = Clone(BO->getRHS());
    } else if (BO->getOpcode() == BO_SubAssign) {
      IsIncrement = false;
      Stride = Clone(BO->getRHS());
    } else if (BO->getOpcode() == BO_Assign) {
      // var = var + incr or var = var - incr
      if (const auto* InnerBO = dyn_cast<BinaryOperator>(BO->getRHS())) {
        if (InnerBO->getOpcode() == BO_Add) {
          IsIncrement = true;
          // Check which side is the variable
          if (const auto* DRE = dyn_cast<DeclRefExpr>(InnerBO->getLHS())) {
            if (DRE->getDecl() == LoopVarDecl)
              Stride = Clone(InnerBO->getRHS());
          } else if (const auto* DRE =
                         dyn_cast<DeclRefExpr>(InnerBO->getRHS())) {
            if (DRE->getDecl() == LoopVarDecl)
              Stride = Clone(InnerBO->getLHS());
          }
        } else if (InnerBO->getOpcode() == BO_Sub) {
          IsIncrement = false;
          if (const auto* DRE = dyn_cast<DeclRefExpr>(InnerBO->getLHS())) {
            if (DRE->getDecl() == LoopVarDecl)
              Stride = Clone(InnerBO->getRHS());
          }
        }
      }
    }
  }

  assert(LoopVarDecl && LowerBound && UpperBound && Stride);

  // If stride is negative (decrement), negate it for getStaticSchedule
  if (!IsIncrement)
    Stride = BuildOp(UO_Minus, Stride);

  // Create variables for chunk bounds: threadlo, threadhi
  QualType IntTy = m_Context.IntTy;
  IdentifierInfo* ThreadLoII = CreateUniqueIdentifier("_t_chunklo");
  VarDecl* ThreadLoDecl = BuildVarDecl(IntTy, ThreadLoII, getZeroInit(IntTy));

  IdentifierInfo* ThreadHiII = CreateUniqueIdentifier("_t_chunkhi");
  VarDecl* ThreadHiDecl = BuildVarDecl(IntTy, ThreadHiII, getZeroInit(IntTy));

  // Build call to GetStaticSchedule(lo, hi, stride, &threadlo, &threadhi)
  llvm::SmallVector<Expr*, 5> ScheduleCallArgs;
  ScheduleCallArgs.push_back(LowerBound);
  ScheduleCallArgs.push_back(UpperBound);
  ScheduleCallArgs.push_back(Stride);
  ScheduleCallArgs.push_back(BuildOp(UO_AddrOf, BuildDeclRef(ThreadLoDecl)));
  ScheduleCallArgs.push_back(BuildOp(UO_AddrOf, BuildDeclRef(ThreadHiDecl)));

  Expr* ScheduleCall =
      GetFunctionCall("GetStaticSchedule", "clad", ScheduleCallArgs);

  // Create forward sweep loop: for (i = threadlo; i <= threadhi; i += stride)
  // Use unique identifier to avoid conflicts when differentiating multiple
  // times
  IdentifierInfo* FwdLoopVarII =
      CreateUniqueIdentifier(LoopVarDecl->getNameAsString());
  VarDecl* FwdLoopVar = BuildVarDecl(LoopVarDecl->getType(), FwdLoopVarII,
                                     BuildDeclRef(ThreadLoDecl));

  Stmt* FwdInit = BuildDeclStmt(FwdLoopVar);

  // Condition: i <= threadhi (or appropriate comparison based on original)
  Expr* FwdCond =
      BuildOp(BO_LE, BuildDeclRef(FwdLoopVar), BuildDeclRef(ThreadHiDecl));

  // Increment: i += stride
  Expr* FwdInc = BuildOp(BO_AddAssign, BuildDeclRef(FwdLoopVar), Clone(Stride));

  // Register loop variable replacement for Visit
  // This ensures references to the original loop variable in the body
  // are replaced with the new forward loop variable
  m_DeclReplacements[LoopVarDecl] = FwdLoopVar;

  // Differentiate the loop body for forward sweep
  StmtDiff BodyDiff = Visit(Body);

  // Clear the replacement after Visit
  m_DeclReplacements.erase(LoopVarDecl);

  // Build forward loop
  Stmt* ForwardLoop =
      new (m_Context) ForStmt(m_Context, FwdInit, FwdCond, nullptr, FwdInc,
                              BodyDiff.getStmt(), noLoc, noLoc, noLoc);

  // For reverse loop, we need to create all variables first, then Visit
  // Create reverse chunk variables
  VarDecl* RevThreadLoDecl = BuildVarDecl(
      IntTy, CreateUniqueIdentifier("_t_chunklo"), getZeroInit(IntTy));
  VarDecl* RevThreadHiDecl = BuildVarDecl(
      IntTy, CreateUniqueIdentifier("_t_chunkhi"), getZeroInit(IntTy));

  // Create reverse loop variable
  IdentifierInfo* RevLoopVarII =
      CreateUniqueIdentifier(LoopVarDecl->getNameAsString());
  VarDecl* RevLoopVar = BuildVarDecl(LoopVarDecl->getType(), RevLoopVarII,
                                     BuildDeclRef(RevThreadHiDecl));

  // For reverse loop, Visit again with the reverse loop variable
  Stmt* ReverseLoopBody = nullptr;
  if (BodyDiff.getStmt_dx()) {
    // Register the reverse loop variable replacement
    m_DeclReplacements[LoopVarDecl] = RevLoopVar;

    // Visit the body again to get the reverse sweep with correct variable refs
    StmtDiff RevBodyDiff = Visit(Body);
    ReverseLoopBody = RevBodyDiff.getStmt_dx();

    // Clear the replacement
    m_DeclReplacements.erase(LoopVarDecl);
  }

  // Create compound statement with declarations and loops
  // Forward: { int threadlo = 0, threadhi = 0;
  //            GetStaticSchedule(...);
  //            for (...) {...} }
  beginBlock(direction::forward);
  addToCurrentBlock(BuildDeclStmt(ThreadLoDecl));
  addToCurrentBlock(BuildDeclStmt(ThreadHiDecl));
  addToCurrentBlock(ScheduleCall);
  addToCurrentBlock(ForwardLoop);
  Stmt* ForwardBlock = endBlock(direction::forward);

  // Reverse: { int threadlo = 0, threadhi = 0;
  //            GetStaticSchedule(...);
  //            for (...) {...} }
  Stmt* ReverseBlock = nullptr;
  if (ReverseLoopBody) {
    // Use the chunk variables we already created
    llvm::SmallVector<Expr*, 5> RevScheduleCallArgs;
    RevScheduleCallArgs.push_back(Clone(LowerBound));
    RevScheduleCallArgs.push_back(Clone(UpperBound));
    RevScheduleCallArgs.push_back(Clone(Stride));
    RevScheduleCallArgs.push_back(
        BuildOp(UO_AddrOf, BuildDeclRef(RevThreadLoDecl)));
    RevScheduleCallArgs.push_back(
        BuildOp(UO_AddrOf, BuildDeclRef(RevThreadHiDecl)));

    Expr* RevScheduleCall =
        GetFunctionCall("GetStaticSchedule", "clad", RevScheduleCallArgs);

    // Create the reverse loop using the already-created loop variable
    Stmt* RevInit = BuildDeclStmt(RevLoopVar);

    // Condition: i_rev >= threadlo
    Expr* RevCond =
        BuildOp(BO_GE, BuildDeclRef(RevLoopVar), BuildDeclRef(RevThreadLoDecl));

    // Decrement: i_rev -= stride
    Expr* RevInc =
        BuildOp(BO_SubAssign, BuildDeclRef(RevLoopVar), Clone(Stride));

    Stmt* ReverseLoop =
        new (m_Context) ForStmt(m_Context, RevInit, RevCond, nullptr, RevInc,
                                ReverseLoopBody, noLoc, noLoc, noLoc);

    beginBlock(direction::reverse);
    addToCurrentBlock(ReverseLoop, direction::reverse);
    addToCurrentBlock(RevScheduleCall, direction::reverse);
    addToCurrentBlock(BuildDeclStmt(RevThreadHiDecl), direction::reverse);
    addToCurrentBlock(BuildDeclStmt(RevThreadLoDecl), direction::reverse);
    ReverseBlock = endBlock(direction::reverse);
  }

  return {ForwardBlock, ReverseBlock};
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
    const auto* CS = D->getInnermostCapturedStmt()->getCapturedStmt();

    // For reverse mode, we need to create two separate OpenMP regions:
    // one for the forward sweep and one for the reverse sweep.
    // Each region needs its own ActOnOpenMPRegionStart/End to properly
    // track variable captures.

    // Create forward OpenMP region
    Stmt* Forward = nullptr;
    CLAD_COMPAT_CLANG19_SemaOpenMP(m_Sema).ActOnOpenMPRegionStart(
        OMPD_parallel, getCurrentScope());
    StmtDiff ForwardBodyDiff;
    {
      Sema::CompoundScopeRAII CompoundScope(m_Sema);

      if (isOpenMPLoopDirective(D->getDirectiveKind())) {
        const auto* FS = cast<ForStmt>(CS);
        ForwardBodyDiff = DifferentiateCanonicalLoop(FS);
      } else {
        ForwardBodyDiff = Visit(CS);
      }
    }
    Forward = CLAD_COMPAT_CLANG19_SemaOpenMP(m_Sema)
                  .ActOnOpenMPRegionEnd(ForwardBodyDiff.getStmt(), OrigClauses)
                  .get();

    // Create reverse OpenMP region
    // We need to re-differentiate to ensure proper variable capture tracking,
    // but we'll use the reverse body from ForwardBodyDiff
    Stmt* Reverse = nullptr;
    CLAD_COMPAT_CLANG19_SemaOpenMP(m_Sema).ActOnOpenMPRegionStart(
        OMPD_parallel, getCurrentScope());
    // Re-differentiate to track captures in the reverse region
    StmtDiff ReverseBodyDiff;
    {
      Sema::CompoundScopeRAII CompoundScope(m_Sema);
      if (isOpenMPLoopDirective(D->getDirectiveKind())) {
        const auto* FS = cast<ForStmt>(CS);
        ReverseBodyDiff = DifferentiateCanonicalLoop(FS);
      } else {
        ReverseBodyDiff = Visit(CS);
      }
    }
    Reverse =
        CLAD_COMPAT_CLANG19_SemaOpenMP(m_Sema)
            .ActOnOpenMPRegionEnd(ReverseBodyDiff.getStmt_dx(), DiffClauses)
            .get();

    AssociatedSDiff = {Forward, Reverse};
  }
  DeclarationNameInfo DirName;
  OpenMPDirectiveKind CancelRegion = OMPD_unknown;
  return {CLAD_COMPAT_CLANG19_SemaOpenMP(m_Sema)
              .ActOnOpenMPExecutableDirective(
                  OMPD_parallel, DirName, CancelRegion, OrigClauses,
                  AssociatedSDiff.getStmt(), D->getBeginLoc(), D->getEndLoc())
              .get(),
          CLAD_COMPAT_CLANG19_SemaOpenMP(m_Sema)
              .ActOnOpenMPExecutableDirective(OMPD_parallel, DirName,
                                              CancelRegion, DiffClauses,
                                              AssociatedSDiff.getStmt_dx(),
                                              D->getBeginLoc(), D->getEndLoc())
              .get()};
}
StmtDiff ReverseModeVisitor::VisitOMPParallelForDirective(
    const clang::OMPParallelForDirective* D) {
  DeclarationNameInfo DirName;
  CLAD_COMPAT_CLANG19_SemaOpenMP(m_Sema).StartOpenMPDSABlock(
      OMPD_parallel, DirName, getCurrentScope(), D->getBeginLoc());
  StmtDiff SDiff = VisitOMPExecutableDirective(D);
  CLAD_COMPAT_CLANG19_SemaOpenMP(m_Sema).EndOpenMPDSABlock(SDiff.getStmt());
  return SDiff;
}
} // namespace clad