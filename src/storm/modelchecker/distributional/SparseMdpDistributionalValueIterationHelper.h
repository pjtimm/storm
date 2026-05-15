#pragma once

#include <cstdint>
#include <vector>

#include "storm/modelchecker/distributional/DistributionalValueIterationOptions.h"
#include "storm/modelchecker/distributional/RewardDistribution.h"
#include "storm/storage/BitVector.h"
#include "storm/storage/SparseMatrix.h"

namespace storm {
namespace modelchecker {
namespace distributional {

template<typename ValueType>
class SparseMdpDistributionalValueIterationHelper {
   public:
    SparseMdpDistributionalValueIterationHelper(storm::storage::SparseMatrix<ValueType> const& transitionMatrix,
                                                std::vector<ValueType> const& stateActionRewards,
                                                storm::storage::BitVector const& targetStates, storm::storage::BitVector const& properStates,
                                                DistributionalValueIterationOptions const& options);

    std::vector<ValueType> computeExpectedRewards();

   private:
    using Distribution = RewardDistribution<ValueType>;

    bool choiceStaysInProperStates(uint64_t choice) const;
    Distribution buildChoiceDistribution(uint64_t choice, std::vector<Distribution> const& distributions) const;
    ValueType computeCategoricalDistance(Distribution const& first, Distribution const& second) const;
    uint64_t getChoiceRewardAsInteger(uint64_t choice) const;

    storm::storage::SparseMatrix<ValueType> const& transitionMatrix;
    std::vector<ValueType> const& stateActionRewards;
    storm::storage::BitVector targetStates;
    storm::storage::BitVector properStates;
    storm::storage::BitVector properNonTargetStates;
    DistributionalValueIterationOptions options;
};

}  // namespace distributional
}  // namespace modelchecker
}  // namespace storm
