#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "storm/exceptions/InvalidArgumentException.h"
#include "storm/modelchecker/distributional/DistributionalValueIterationOptions.h"
#include "storm/modelchecker/distributional/RewardDistribution.h"
#include "storm/modelchecker/distributional/SparseMdpCvarPreprocessor.h"
#include "storm/storage/BitVector.h"
#include "storm/storage/SparseMatrix.h"
#include "storm/utility/macros.h"

namespace storm {
namespace modelchecker {
namespace distributional {

template<typename ValueType>
class SparseMdpCvarObjective {
   public:
    using Distribution = RewardDistribution<ValueType>;
    using PreprocessorResult = typename SparseMdpCvarPreprocessor<ValueType>::Result;

    class ProductDistributionCache {
       public:
        uint64_t getCachedDistributionCount() const {
            return distributions.size();
        }

        bool hasDistribution(uint64_t productState) const {
            return distributions.find(productState) != distributions.end();
        }

        Distribution const& getDistribution(uint64_t productState) const {
            auto const it = distributions.find(productState);
            STORM_LOG_THROW(it != distributions.end(), storm::exceptions::InvalidArgumentException,
                            "No cached CVaR product distribution exists for product state " << productState << ".");
            return it->second;
        }

       private:
        friend class SparseMdpCvarObjective<ValueType>;

        std::unordered_map<uint64_t, Distribution> distributions;
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
    ProductDistributionCache createProductDistributionCache() const;
    Distribution const& getOrInitializeProductDistribution(ProductDistributionCache& cache, uint64_t productState) const;

   private:
    void validateDimensions() const;

    storm::storage::SparseMatrix<ValueType> const& transitionMatrix;
    std::vector<ValueType> const& stateActionRewards;
    storm::storage::BitVector targetStates;
    storm::storage::BitVector properStates;
    DistributionalValueIterationOptions options;
    PreprocessorResult const& preprocessorResult;
    uint64_t stateCount;
    uint64_t budgetCount;
};

}  // namespace distributional
}  // namespace modelchecker
}  // namespace storm
