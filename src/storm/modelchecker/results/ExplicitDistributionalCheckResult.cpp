#include "storm/modelchecker/results/ExplicitDistributionalCheckResult.h"

#include "storm/adapters/RationalNumberAdapter.h"
#include "storm/exceptions/InvalidAccessException.h"
#include "storm/exceptions/InvalidOperationException.h"
#include "storm/modelchecker/results/ExplicitQualitativeCheckResult.h"
#include "storm/utility/constants.h"
#include "storm/utility/macros.h"

namespace storm {
namespace modelchecker {

template<typename ValueType>
ExplicitDistributionalCheckResult<ValueType>::ExplicitDistributionalCheckResult(distribution_vector_type&& distributions,
                                                                                storm::storage::BitVector&& finiteDistributionStates)
    : distributions(std::move(distributions)), finiteDistributionStates(std::move(finiteDistributionStates)), resultStates() {
    STORM_LOG_THROW(this->finiteDistributionStates.size() == this->getDistributionVector().size(), storm::exceptions::InvalidOperationException,
                    "Distributional result has an invalid finite-state vector.");
}

template<typename ValueType>
ExplicitDistributionalCheckResult<ValueType>::ExplicitDistributionalCheckResult(distribution_map_type&& distributions)
    : distributions(std::move(distributions)), finiteDistributionStates(), resultStates() {
    for (auto const& entry : getDistributionMap()) {
        resultStates.insert(entry.first);
    }
}

template<typename ValueType>
std::unique_ptr<CheckResult> ExplicitDistributionalCheckResult<ValueType>::clone() const {
    return std::make_unique<ExplicitDistributionalCheckResult<ValueType>>(*this);
}

template<typename ValueType>
bool ExplicitDistributionalCheckResult<ValueType>::isExplicit() const {
    return true;
}

template<typename ValueType>
bool ExplicitDistributionalCheckResult<ValueType>::isResultForAllStates() const {
    return distributions.type() == typeid(distribution_vector_type);
}

template<typename ValueType>
bool ExplicitDistributionalCheckResult<ValueType>::isExplicitDistributionalCheckResult() const {
    return true;
}

template<typename ValueType>
typename ExplicitDistributionalCheckResult<ValueType>::distribution_vector_type const& ExplicitDistributionalCheckResult<ValueType>::getDistributionVector()
    const {
    return boost::get<distribution_vector_type>(distributions);
}

template<typename ValueType>
typename ExplicitDistributionalCheckResult<ValueType>::distribution_map_type const& ExplicitDistributionalCheckResult<ValueType>::getDistributionMap() const {
    return boost::get<distribution_map_type>(distributions);
}

template<typename ValueType>
bool ExplicitDistributionalCheckResult<ValueType>::hasFiniteDistribution(storm::storage::sparse::state_type state) const {
    if (isResultForAllStates()) {
        return state < finiteDistributionStates.size() && finiteDistributionStates.get(state);
    }
    return getDistributionMap().find(state) != getDistributionMap().end();
}

template<typename ValueType>
bool ExplicitDistributionalCheckResult<ValueType>::hasDistribution(storm::storage::sparse::state_type state) const {
    return hasFiniteDistribution(state);
}

template<typename ValueType>
typename ExplicitDistributionalCheckResult<ValueType>::distribution_type const& ExplicitDistributionalCheckResult<ValueType>::getDistribution(
    storm::storage::sparse::state_type state) const {
    STORM_LOG_THROW(hasFiniteDistribution(state), storm::exceptions::InvalidAccessException, "No finite distribution stored for state " << state << ".");
    if (isResultForAllStates()) {
        return getDistributionVector()[state];
    }
    return getDistributionMap().at(state);
}

template<typename ValueType>
ValueType ExplicitDistributionalCheckResult<ValueType>::getExpectedValue(storm::storage::sparse::state_type state) const {
    return getDistribution(state).getProjectedExpectedValue();
}

template<typename ValueType>
void ExplicitDistributionalCheckResult<ValueType>::filter(QualitativeCheckResult const& filter) {
    STORM_LOG_THROW(filter.isExplicitQualitativeCheckResult(), storm::exceptions::InvalidOperationException,
                    "Cannot filter explicit distributional check result with non-explicit filter.");
    STORM_LOG_THROW(filter.isResultForAllStates(), storm::exceptions::InvalidOperationException,
                    "Cannot filter distributional check result with non-complete filter.");
    STORM_LOG_THROW(filter.hasValueType<ValueType>(), storm::exceptions::InvalidOperationException, "Filter has unexpected value type.");

    auto const& explicitFilter = filter.template asExplicitQualitativeCheckResult<ValueType>();
    auto const& filterTruthValues = explicitFilter.getTruthValuesVector();

    distribution_map_type newDistributions;
    state_set_type newResultStates;
    if (isResultForAllStates()) {
        auto const& distributionVector = getDistributionVector();
        for (auto state : filterTruthValues) {
            STORM_LOG_THROW(state < distributionVector.size(), storm::exceptions::InvalidAccessException, "Invalid index in distributional result.");
            newResultStates.insert(state);
            if (finiteDistributionStates.get(state)) {
                newDistributions.emplace(state, distributionVector[state]);
            }
        }
    } else {
        auto const& distributionMap = getDistributionMap();
        for (auto const state : resultStates) {
            if (filterTruthValues.get(state)) {
                newResultStates.insert(state);
                auto distributionIt = distributionMap.find(state);
                if (distributionIt != distributionMap.end()) {
                    newDistributions.insert(*distributionIt);
                }
            }
        }
    }

    distributions = std::move(newDistributions);
    resultStates = std::move(newResultStates);
    finiteDistributionStates = storm::storage::BitVector();
}

template<typename ValueType>
std::ostream& ExplicitDistributionalCheckResult<ValueType>::writeToStream(std::ostream& out) const {
    auto printDistribution = [&out](distribution_type const& distribution) {
        out << "{";
        bool first = true;
        distribution.forEachMass([&out, &first](ValueType const& reward, ValueType const& mass) {
            if (storm::utility::isZero(mass)) {
                return;
            }
            if (!first) {
                out << ", ";
            }
            first = false;
            out << reward << ": " << mass;
        });
        out << "}";
    };

    if (isResultForAllStates()) {
        auto const& distributionVector = getDistributionVector();
        if (distributionVector.size() >= 10) {
            out << distributionVector.size() << " distributional values";
        } else {
            out << "{";
            bool first = true;
            for (uint64_t state = 0; state < distributionVector.size(); ++state) {
                if (!first) {
                    out << ", ";
                }
                first = false;
                if (hasFiniteDistribution(state)) {
                    printDistribution(getDistribution(state));
                } else {
                    out << "inf";
                }
            }
            out << "}";
        }
    } else {
        if (resultStates.size() == 1) {
            auto const state = *resultStates.begin();
            if (hasFiniteDistribution(state)) {
                printDistribution(getDistribution(state));
            } else {
                out << "inf";
            }
        } else {
            out << "{";
            bool first = true;
            for (auto const state : resultStates) {
                if (!first) {
                    out << ", ";
                }
                first = false;
                out << state << ": ";
                if (hasFiniteDistribution(state)) {
                    printDistribution(getDistribution(state));
                } else {
                    out << "inf";
                }
            }
            out << "}";
        }
    }
    return out;
}

template class ExplicitDistributionalCheckResult<double>;
template class ExplicitDistributionalCheckResult<storm::RationalNumber>;

}  // namespace modelchecker
}  // namespace storm
