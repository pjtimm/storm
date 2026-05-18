#include "storm/modelchecker/distributional/SparseMdpDistributionalViHelper.h"

#include <utility>

#include "storm/adapters/RationalNumberAdapter.h"
#include "storm/exceptions/InvalidArgumentException.h"
#include "storm/exceptions/UnexpectedException.h"
#include "storm/utility/constants.h"
#include "storm/utility/macros.h"

namespace storm {
namespace modelchecker {
namespace distributional {

template<typename ValueType>
SparseMdpDistributionalViHelper<ValueType>::SparseMdpDistributionalViHelper(storm::storage::SparseMatrix<ValueType> const& transitionMatrix,
                                                                            std::vector<ValueType> const& stateActionRewards,
                                                                            storm::storage::BitVector const& targetStates,
                                                                            storm::storage::BitVector const& properStates,
                                                                            DistributionalValueIterationOptions const& options)
    : transitionMatrix(transitionMatrix),
      stateActionRewards(stateActionRewards),
      targetStates(targetStates),
      properStates(properStates),
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
bool SparseMdpDistributionalViHelper<ValueType>::isChoiceAdmissible(uint64_t choice) const {
    return admissibleChoices.get(choice);
}

template<typename ValueType>
typename SparseMdpDistributionalViHelper<ValueType>::Distribution SparseMdpDistributionalViHelper<ValueType>::buildChoiceDistribution(
    uint64_t choice, std::vector<Distribution> const& distributions) const {
    RewardDistributionBuilder<ValueType> builder(getRewardDistributionOptions());
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
ValueType SparseMdpDistributionalViHelper<ValueType>::computeCategoricalSquaredDistance(Distribution const& first, Distribution const& second) const {
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
uint64_t SparseMdpDistributionalViHelper<ValueType>::getChoiceRewardAsInteger(uint64_t choice) const {
    return storm::utility::convertNumber<uint64_t, ValueType>(stateActionRewards[choice]);
}

template<typename ValueType>
RewardDistributionOptions SparseMdpDistributionalViHelper<ValueType>::getRewardDistributionOptions() const {
    return options.toRewardDistributionOptions();
}

template<typename ValueType>
bool SparseMdpDistributionalViHelper<ValueType>::choiceStaysInProperStates(uint64_t choice) const {
    for (auto const& entry : transitionMatrix.getRow(choice)) {
        if (!storm::utility::isZero(entry.getValue()) && !targetStates.get(entry.getColumn()) && !properStates.get(entry.getColumn())) {
            return false;
        }
    }
    return true;
}

template class SparseMdpDistributionalViHelper<double>;
template class SparseMdpDistributionalViHelper<storm::RationalNumber>;

}  // namespace distributional
}  // namespace modelchecker
}  // namespace storm
