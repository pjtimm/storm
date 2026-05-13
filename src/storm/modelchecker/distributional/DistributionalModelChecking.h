#pragma once

#include <memory>

#include "storm/logic/FormulasForwardDeclarations.h"

namespace storm {
class Environment;

namespace modelchecker {
class CheckResult;

namespace distributional {

template<typename SparseModelType>
std::unique_ptr<CheckResult> performDistributionalModelChecking(Environment const& env, SparseModelType const& model,
                                                                storm::logic::DistributionalFormula const& formula, bool produceScheduler = false);

}  // namespace distributional
}  // namespace modelchecker
}  // namespace storm
