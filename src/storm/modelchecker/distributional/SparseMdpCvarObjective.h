#pragma once

#include <cstdint>
#include <map>
#include <unordered_map>
#include <vector>

#include "storm/modelchecker/distributional/DistributionalValueIterationOptions.h"
#include "storm/modelchecker/distributional/RewardDistribution.h"
#include "storm/modelchecker/distributional/SparseMdpDistributionalViHelper.h"
#include "storm/modelchecker/distributional/SparseMdpCvarPreprocessor.h"
#include "storm/storage/BitVector.h"
#include "storm/storage/SparseMatrix.h"
#include "storm/storage/sparse/StateType.h"

namespace storm {
namespace modelchecker {
namespace distributional {

template<typename ValueType>
class SparseMdpCvarObjective {
   public:
    using Distribution = RewardDistribution<ValueType>;
    using DistributionMap = std::map<storm::storage::sparse::state_type, Distribution>;
    using PreprocessorResult = typename SparseMdpCvarPreprocessor<ValueType>::Result;

    struct Result {
        DistributionMap distributions;
    };

    SparseMdpCvarObjective(storm::storage::SparseMatrix<ValueType> const& transitionMatrix, std::vector<ValueType> const& stateActionRewards,
                           storm::storage::BitVector const& targetStates, storm::storage::BitVector const& properStates,
                           DistributionalValueIterationOptions const& options, PreprocessorResult const& preprocessorResult);

    uint64_t getStateCount() const;
    uint64_t getBudgetCount() const;
    uint64_t getProductStateCount() const;

    uint64_t getProductStateIndex(uint64_t state, uint64_t budgetIndex) const;
    uint64_t getOriginalState(uint64_t productState) const;
    uint64_t getBudgetIndex(uint64_t productState) const;

    bool isFiniteProductState(uint64_t productState) const;
    Result computeCvarOptimalDistribution() const;

   private:
    struct ReachableProductStates {
        std::vector<uint64_t> states;
        std::unordered_map<uint64_t, uint64_t> indices;
        std::vector<std::vector<uint64_t>> indicesByState;
    };

    void validateDimensions() const;
    bool isChoiceAdmissible(uint64_t choice) const;
    uint64_t getChoiceRewardAsInteger(uint64_t choice) const;
    Distribution buildProductChoiceDistribution(ReachableProductStates const& productStates, std::vector<Distribution> const& previousDistributions,
                                                uint64_t choice, uint64_t budgetIndex) const;
    ValueType computeTailExpectation(Distribution const& distribution, ValueType const& budget) const;
    ReachableProductStates computeReachableProductStates() const;
    void runTopologicalViSweep(ReachableProductStates const& productStates, std::vector<Distribution>& distributions) const;
    Result selectInitialDistribution(ReachableProductStates const& productStates, std::vector<Distribution> const& distributions) const;

    storm::storage::SparseMatrix<ValueType> const& transitionMatrix;
    std::vector<ValueType> const& stateActionRewards;
    storm::storage::BitVector targetStates;
    storm::storage::BitVector properStates;
    DistributionalValueIterationOptions options;
    SparseMdpDistributionalViHelper<ValueType> viHelper;
    PreprocessorResult const& preprocessorResult;
    uint64_t stateCount;
    uint64_t budgetCount;
};

}  // namespace distributional
}  // namespace modelchecker
}  // namespace storm
