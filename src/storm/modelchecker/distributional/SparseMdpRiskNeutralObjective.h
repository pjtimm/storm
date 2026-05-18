#pragma once

#include <cstdint>
#include <vector>

#include "storm/modelchecker/distributional/DistributionalValueIterationOptions.h"
#include "storm/modelchecker/distributional/RewardDistribution.h"
#include "storm/modelchecker/distributional/SparseMdpDistributionalViHelper.h"
#include "storm/storage/BitVector.h"
#include "storm/storage/SparseMatrix.h"

namespace storm {
namespace modelchecker {
namespace distributional {

template<typename ValueType>
class SparseMdpRiskNeutralObjective {
   public:
    using Distribution = RewardDistribution<ValueType>;

    struct Result {
        std::vector<Distribution> distributions;
        storm::storage::BitVector finiteDistributionStates;
    };

    SparseMdpRiskNeutralObjective(storm::storage::SparseMatrix<ValueType> const& transitionMatrix, std::vector<ValueType> const& stateActionRewards,
                                  storm::storage::BitVector const& targetStates, storm::storage::BitVector const& properStates,
                                  DistributionalValueIterationOptions const& options);

    Result computeExpectedRewardOptimalDistributions();

   private:
    storm::storage::SparseMatrix<ValueType> const& transitionMatrix;
    storm::storage::BitVector targetStates;
    storm::storage::BitVector properStates;
    storm::storage::BitVector properNonTargetStates;
    DistributionalValueIterationOptions options;
    SparseMdpDistributionalViHelper<ValueType> viHelper;
};

}  // namespace distributional
}  // namespace modelchecker
}  // namespace storm
