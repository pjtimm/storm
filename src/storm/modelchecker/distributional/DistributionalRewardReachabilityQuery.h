#pragma once

#include "storm/exceptions/InvalidPropertyException.h"
#include "storm/logic/Formulas.h"
#include "storm/utility/macros.h"

namespace storm {
namespace modelchecker {
namespace distributional {

struct DistributionalRewardReachabilityQuery {
    storm::logic::RewardOperatorFormula const& rewardOperatorFormula;
    storm::logic::Formula const& targetFormula;
};

inline DistributionalRewardReachabilityQuery parseDistributionalRewardReachabilityQuery(storm::logic::DistributionalFormula const& formula) {
    storm::logic::Formula const& subformula = formula.getSubformula();
    STORM_LOG_THROW(subformula.isRewardOperatorFormula(), storm::exceptions::InvalidPropertyException,
                    "Distributional model checking currently requires a reward operator formula, but got '" << subformula << "'.");

    storm::logic::RewardOperatorFormula const& rewardOperatorFormula = subformula.asRewardOperatorFormula();
    STORM_LOG_THROW(rewardOperatorFormula.hasQuantitativeResult(), storm::exceptions::InvalidPropertyException,
                    "Distributional model checking currently requires a quantitative reward query without a comparison bound.");
    STORM_LOG_THROW(
        rewardOperatorFormula.getSubformula().isReachabilityRewardFormula(), storm::exceptions::InvalidPropertyException,
        "Distributional model checking currently requires a reachability reward formula, but got '" << rewardOperatorFormula.getSubformula() << "'.");

    storm::logic::EventuallyFormula const& reachabilityRewardFormula = rewardOperatorFormula.getSubformula().asReachabilityRewardFormula();
    STORM_LOG_THROW(reachabilityRewardFormula.getSubformula().isStateFormula(), storm::exceptions::InvalidPropertyException,
                    "Distributional model checking currently requires a state target formula.");
    return DistributionalRewardReachabilityQuery{rewardOperatorFormula, reachabilityRewardFormula.getSubformula()};
}

}  // namespace distributional
}  // namespace modelchecker
}  // namespace storm
