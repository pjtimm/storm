#pragma once

#include "storm/solver/OptimizationDirection.h"
#include "storm/utility/macros.h"

namespace storm {
namespace modelchecker {
namespace distributional {

enum class DistributionalCvarInterpretation { Cost, Reward };

enum class DistributionalCvarInterpretationSelection { Auto, Cost, Reward };

inline DistributionalCvarInterpretation resolveDistributionalCvarInterpretation(DistributionalCvarInterpretationSelection selection,
                                                                                storm::solver::OptimizationDirection optimizationDirection) {
    switch (selection) {
        case DistributionalCvarInterpretationSelection::Auto:
            return storm::solver::minimize(optimizationDirection) ? DistributionalCvarInterpretation::Cost : DistributionalCvarInterpretation::Reward;
        case DistributionalCvarInterpretationSelection::Cost:
            return DistributionalCvarInterpretation::Cost;
        case DistributionalCvarInterpretationSelection::Reward:
            return DistributionalCvarInterpretation::Reward;
    }
    STORM_LOG_ASSERT(false, "Encountered an unknown distributional CVaR interpretation selection.");
    return DistributionalCvarInterpretation::Cost;
}

}  // namespace distributional
}  // namespace modelchecker
}  // namespace storm
