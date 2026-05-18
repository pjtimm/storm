#pragma once

#include <boost/variant.hpp>
#include <map>
#include <set>
#include <vector>

#include "storm/modelchecker/distributional/RewardDistribution.h"
#include "storm/modelchecker/results/CheckResult.h"
#include "storm/storage/BitVector.h"
#include "storm/storage/sparse/StateType.h"

namespace storm {
namespace modelchecker {

template<typename ValueType>
class ExplicitDistributionalCheckResult : public CheckResult {
   public:
    using distribution_type = storm::modelchecker::distributional::RewardDistribution<ValueType>;
    using distribution_vector_type = std::vector<distribution_type>;
    using distribution_map_type = std::map<storm::storage::sparse::state_type, distribution_type>;
    using state_set_type = std::set<storm::storage::sparse::state_type>;

    ExplicitDistributionalCheckResult(distribution_vector_type&& distributions, storm::storage::BitVector&& finiteDistributionStates);
    ExplicitDistributionalCheckResult(distribution_map_type&& distributions);
    ExplicitDistributionalCheckResult(ExplicitDistributionalCheckResult const& other) = default;
    ExplicitDistributionalCheckResult& operator=(ExplicitDistributionalCheckResult const& other) = default;
    ExplicitDistributionalCheckResult(ExplicitDistributionalCheckResult&& other) = default;
    ExplicitDistributionalCheckResult& operator=(ExplicitDistributionalCheckResult&& other) = default;
    virtual ~ExplicitDistributionalCheckResult() = default;

    virtual std::unique_ptr<CheckResult> clone() const override;

    virtual bool isExplicit() const override;
    virtual bool isResultForAllStates() const override;
    virtual bool isExplicitDistributionalCheckResult() const override;

    distribution_vector_type const& getDistributionVector() const;
    distribution_map_type const& getDistributionMap() const;

    bool hasDistribution(storm::storage::sparse::state_type state) const;
    distribution_type const& getDistribution(storm::storage::sparse::state_type state) const;
    ValueType getExpectedValue(storm::storage::sparse::state_type state) const;

    virtual void filter(QualitativeCheckResult const& filter) override;
    virtual std::ostream& writeToStream(std::ostream& out) const override;

   private:
    bool hasValueType(std::type_info const& t) const override {
        return t == typeid(ValueType);
    }

    boost::variant<distribution_vector_type, distribution_map_type> distributions;
    storm::storage::BitVector finiteDistributionStates;
    state_set_type resultStates;
};

}  // namespace modelchecker
}  // namespace storm
