#include "clad/Differentiator/MultiplexExternalRMVSource.h"
#include "clad/Differentiator/ReverseModeVisitor.h"
#include <clang/AST/OpenMPClause.h>
#include <clang/AST/StmtOpenMP.h>
#include <clang/Basic/OpenMPKinds.h>
#include <llvm/Frontend/OpenMP/OMP.h.inc>
#include <llvm/Support/ErrorHandling.h>

#include <numeric>

using namespace clang;
using namespace llvm::omp;

namespace clad {
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