#include "storm/modelchecker/distributional/SparseMdpDistributionalValueIterationHelper.h"

#include <algorithm>

#include <boost/optional.hpp>

#include "storm/adapters/RationalNumberAdapter.h"
#include "storm/exceptions/InvalidArgumentException.h"
#include "storm/exceptions/NoConvergenceException.h"
#include "storm/exceptions/UnexpectedException.h"
#include "storm/utility/constants.h"
#include "storm/utility/macros.h"

namespace storm {
namespace modelchecker {
namespace distributional {

template<typename ValueType>
SparseMdpDistributionalValueIterationHelper<ValueType>::SparseMdpDistributionalValueIterationHelper(
    storm::storage::SparseMatrix<ValueType> const& transitionMatrix, std::vector<ValueType> const& stateActionRewards,
    storm::storage::BitVector const& targetStates, storm::storage::BitVector const& properStates, DistributionalValueIterationOptions const& options)
    : transitionMatrix(transitionMatrix),
      stateActionRewards(stateActionRewards),
      targetStates(targetStates),
      properStates(properStates),
      properNonTargetStates(properStates & ~targetStates),
      options(options),
      admissibleChoices(transitionMatrix.getRowCount(), false) {
    STORM_LOG_THROW(transitionMatrix.getRowCount() == stateActionRewards.size(), storm::exceptions::InvalidArgumentException,
                    "Expected one distributional reward value per nondeterministic choice.");
    STORM_LOG_THROW(transitionMatrix.getRowGroupCount() == targetStates.size(), storm::exceptions::InvalidArgumentException,
                    "Distributional target-state vector has unexpected size.");
    STORM_LOG_THROW(transitionMatrix.getRowGroupCount() == properStates.size(), storm::exceptions::InvalidArgumentException,
                    "Distributional proper-state vector has unexpected size.");
    for (uint64_t choice = 0; choice < transitionMatrix.getRowCount(); ++choice) {
        admissibleChoices.set(choice, choiceStaysInProperStates(choice));
    }
}

template<typename ValueType>
typename SparseMdpDistributionalValueIterationHelper<ValueType>::Result
SparseMdpDistributionalValueIterationHelper<ValueType>::computeExpectedRewardOptimalDistributions() {
    RewardDistributionOptions const rewardDistributionOptions = options.toRewardDistributionOptions();
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
                if (!admissibleChoices.get(choice)) {
                    continue;
                }
                Distribution choiceDistribution = buildChoiceDistribution(choice, distributions);
                ValueType choiceExpectation = choiceDistribution.getProjectedExpectedValue();
                if (!bestDistribution || choiceExpectation < bestExpectation) {
                    bestExpectation = choiceExpectation;
                    bestDistribution = std::move(choiceDistribution);
                }
            }
            STORM_LOG_THROW(bestDistribution, storm::exceptions::UnexpectedException,
                            "Expected at least one admissible distributional choice for a proper state.");
            maximalSquaredDistance = std::max(maximalSquaredDistance, computeCategoricalSquaredDistance(distributions[state], bestDistribution.get()));
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

template<typename ValueType>
bool SparseMdpDistributionalValueIterationHelper<ValueType>::choiceStaysInProperStates(uint64_t choice) const {
    for (auto const& entry : transitionMatrix.getRow(choice)) {
        if (!storm::utility::isZero(entry.getValue()) && !targetStates.get(entry.getColumn()) && !properStates.get(entry.getColumn())) {
            return false;
        }
    }
    return true;
}

template<typename ValueType>
typename SparseMdpDistributionalValueIterationHelper<ValueType>::Distribution SparseMdpDistributionalValueIterationHelper<ValueType>::buildChoiceDistribution(
    uint64_t choice, std::vector<Distribution> const& distributions) const {
    RewardDistributionBuilder<ValueType> builder(options.toRewardDistributionOptions());
    uint64_t const reward = getChoiceRewardAsInteger(choice);
    for (auto const& entry : transitionMatrix.getRow(choice)) {
        if (storm::utility::isZero(entry.getValue())) {
            continue;
        }
        builder.addScaledShifted(entry.getValue(), distributions[entry.getColumn()], reward);
    }
    return std::move(builder).build();
}

template<typename ValueType>
ValueType SparseMdpDistributionalValueIterationHelper<ValueType>::computeCategoricalSquaredDistance(Distribution const& first,
                                                                                                    Distribution const& second) const {
    STORM_LOG_THROW(first.isCategorical() && second.isCategorical(), storm::exceptions::UnexpectedException,
                    "Expected categorical distributions when computing convergence distance.");
    auto const& firstMasses = first.getCategoricalMasses();
    auto const& secondMasses = second.getCategoricalMasses();
    STORM_LOG_THROW(firstMasses.size() == secondMasses.size(), storm::exceptions::UnexpectedException,
                    "Cannot compare categorical distributions with different atom counts.");
    ValueType result = storm::utility::zero<ValueType>();
    ValueType cdfDifference = storm::utility::zero<ValueType>();
    for (uint64_t atom = 0; atom + 1 < firstMasses.size(); ++atom) {
        cdfDifference += firstMasses[atom] - secondMasses[atom];
        result += cdfDifference * cdfDifference;
    }
    ValueType const stepSize = storm::utility::convertNumber<ValueType, uint64_t>(options.stepSize);
    return stepSize * result;
}

template<typename ValueType>
uint64_t SparseMdpDistributionalValueIterationHelper<ValueType>::getChoiceRewardAsInteger(uint64_t choice) const {
    return storm::utility::convertNumber<uint64_t, ValueType>(stateActionRewards[choice]);
}

template class SparseMdpDistributionalValueIterationHelper<double>;
template class SparseMdpDistributionalValueIterationHelper<storm::RationalNumber>;

}  // namespace distributional
}  // namespace modelchecker
}  // namespace storm
