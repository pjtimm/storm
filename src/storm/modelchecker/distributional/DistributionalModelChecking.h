#pragma once

#include <memory>

#include "storm/logic/FormulasForwardDeclarations.h"

namespace storm {
class Environment;

namespace modelchecker {
template<typename FormulaType, typename ValueType>
class CheckTask;
class CheckResult;

namespace distributional {

template<typename SparseModelType, typename SolutionType>
std::unique_ptr<CheckResult> performDistributionalModelChecking(Environment const& env, SparseModelType const& model,
                                                                CheckTask<storm::logic::DistributionalFormula, SolutionType> const& checkTask);

}  // namespace distributional
}  // namespace modelchecker
}  // namespace storm
