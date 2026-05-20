#include "storm/modelchecker/distributional/SparseMdpRiskNeutralObjective.h"

#include <algorithm>
#include <utility>

#include <boost/optional.hpp>

#include "storm/adapters/RationalNumberAdapter.h"
#include "storm/exceptions/NoConvergenceException.h"
#include "storm/exceptions/UnexpectedException.h"
#include "storm/utility/constants.h"
#include "storm/utility/macros.h"

namespace storm {
namespace modelchecker {
namespace distributional {

template<typename ValueType>
SparseMdpRiskNeutralObjective<ValueType>::SparseMdpRiskNeutralObjective(storm::storage::SparseMatrix<ValueType> const& transitionMatrix,
                                                                        std::vector<ValueType> const& stateActionRewards,
                                                                        storm::storage::BitVector const& targetStates,
                                                                        storm::storage::BitVector const& properStates,
                                                                        DistributionalValueIterationOptions const& options)
    : transitionMatrix(transitionMatrix),
      targetStates(targetStates),
      properStates(properStates),
      properNonTargetStates(properStates & ~targetStates),
      options(options),
      viHelper(transitionMatrix, stateActionRewards, targetStates, properStates, options) {
    // Intentionally left empty.
}

template<typename ValueType>
typename SparseMdpRiskNeutralObjective<ValueType>::Result SparseMdpRiskNeutralObjective<ValueType>::computeExpectedRewardOptimalDistributions() {
    RewardDistributionOptions const rewardDistributionOptions = viHelper.getRewardDistributionOptions();
    std::vector<Distribution> distributions(transitionMatrix.getRowGroupCount(), Distribution::categoricalTail(rewardDistributionOptions));
    for (auto const targetState : targetStates) {
        distributions[targetState] = Distribution::pointMass(0);
    }
    std::vector<Distribution> newDistributions = distributions;

    for (uint64_t iteration = 0; iteration < options.maximalIterations; ++iteration) {
        ValueType maximalSquaredDistance = storm::utility::zero<ValueType>();
        for (auto const state : properNonTargetStates) {
            boost::optional<Distribution> bestDistribution;
            ValueType bestExpectation = storm::utility::zero<ValueType>();
            for (auto const choice : transitionMatrix.getRowGroupIndices(state)) {
                if (!viHelper.isChoiceAdmissible(choice)) {
                    continue;
                }
                Distribution choiceDistribution = viHelper.buildChoiceDistribution(choice, distributions);
                ValueType choiceExpectation = choiceDistribution.getProjectedExpectedValue();
                if (!bestDistribution || choiceExpectation < bestExpectation) {
                    bestExpectation = choiceExpectation;
                    bestDistribution = std::move(choiceDistribution);
                }
            }
            STORM_LOG_THROW(bestDistribution, storm::exceptions::UnexpectedException,
                            "Expected at least one admissible distributional choice for a proper state.");
            maximalSquaredDistance = std::max(maximalSquaredDistance, viHelper.computeCategoricalSquaredDistance(distributions[state], bestDistribution.get()));
            newDistributions[state] = std::move(bestDistribution.get());
        }
        distributions.swap(newDistributions);
        ValueType const precision = storm::utility::convertNumber<ValueType>(options.precision);
        if (maximalSquaredDistance <= precision * precision) {
            return Result{std::move(distributions), properStates};
        }
    }

    STORM_LOG_THROW(false, storm::exceptions::NoConvergenceException,
                    "Distributional value iteration did not converge within " << options.maximalIterations << " iterations.");
    return {};
}

template class SparseMdpRiskNeutralObjective<double>;
template class SparseMdpRiskNeutralObjective<storm::RationalNumber>;

}  // namespace distributional
}  // namespace modelchecker
}  // namespace storm
