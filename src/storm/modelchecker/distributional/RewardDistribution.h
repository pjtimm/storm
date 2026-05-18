#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "storm/exceptions/InvalidArgumentException.h"
#include "storm/exceptions/NotSupportedException.h"
#include "storm/modelchecker/distributional/RewardDistributionRepresentation.h"
#include "storm/storage/Distribution.h"
#include "storm/utility/constants.h"
#include "storm/utility/macros.h"

namespace storm {
namespace modelchecker {
namespace distributional {

struct RewardDistributionOptions {
    using Representation = RewardDistributionRepresentation;

    Representation representation = Representation::Categorical;
    uint64_t atoms = 101;
    uint64_t stepSize = 1;

    uint64_t getCategoricalUpperRewardBound() const {
        if (atoms == 0) {
            return 0;
        }
        STORM_LOG_THROW(stepSize > 0, storm::exceptions::InvalidArgumentException, "Categorical reward distributions require a positive reward step size.");
        STORM_LOG_THROW(atoms - 1 <= std::numeric_limits<uint64_t>::max() / stepSize, storm::exceptions::InvalidArgumentException,
                        "Categorical reward grid upper bound exceeds the supported integer range.");
        return (atoms - 1) * stepSize;
    }
};

template<typename ValueType>
class RewardDistribution {
   public:
    using ExactDistribution = storm::storage::Distribution<ValueType, uint64_t>;

    enum class Kind { ExactSparse, Categorical };

    static RewardDistribution pointMass(uint64_t reward) {
        ExactDistribution distribution;
        distribution.addProbability(reward, storm::utility::one<ValueType>());
        return RewardDistribution(std::move(distribution));
    }

    static RewardDistribution categoricalTail(RewardDistributionOptions const& options) {
        STORM_LOG_THROW(options.representation == RewardDistributionOptions::Representation::Categorical, storm::exceptions::InvalidArgumentException,
                        "Can only create a categorical tail distribution for categorical options.");
        STORM_LOG_THROW(options.atoms > 0, storm::exceptions::InvalidArgumentException, "Categorical reward distributions require a positive atom count.");
        std::vector<ValueType> masses(options.atoms, storm::utility::zero<ValueType>());
        masses.back() = storm::utility::one<ValueType>();
        return RewardDistribution(std::move(masses), 0, options.getCategoricalUpperRewardBound());
    }

    Kind getKind() const {
        return kind;
    }

    bool isExact() const {
        return kind == Kind::ExactSparse;
    }

    bool isCategorical() const {
        return kind == Kind::Categorical;
    }

    uint64_t getSupportSize() const {
        return isExact() ? exactMasses.size() : categoricalMasses.size();
    }

    ExactDistribution const& getExactMasses() const {
        STORM_LOG_THROW(isExact(), storm::exceptions::InvalidArgumentException, "Requested exact masses from a non-exact reward distribution.");
        return exactMasses;
    }

    std::vector<ValueType> const& getCategoricalMasses() const {
        STORM_LOG_THROW(isCategorical(), storm::exceptions::InvalidArgumentException,
                        "Requested categorical masses from a non-categorical reward distribution.");
        return categoricalMasses;
    }

    uint64_t getLowerRewardBound() const {
        return lowerRewardBound;
    }

    uint64_t getUpperRewardBound() const {
        return upperRewardBound;
    }

    ValueType getProjectedExpectedValue() const {
        ValueType result = storm::utility::zero<ValueType>();
        forEachMass([&result](ValueType const& reward, ValueType const& mass) { result += reward * mass; });
        return result;
    }

    ValueType getExpectedValue() const {
        return getProjectedExpectedValue();
    }

    template<typename Callback>
    void forEachMass(Callback&& callback) const {
        if (isExact()) {
            for (auto const& entry : exactMasses) {
                callback(storm::utility::convertNumber<ValueType, uint64_t>(entry.first), entry.second);
            }
        } else {
            for (uint64_t atom = 0; atom < categoricalMasses.size(); ++atom) {
                callback(getAtomValue(atom, lowerRewardBound, upperRewardBound, categoricalMasses.size()), categoricalMasses[atom]);
            }
        }
    }

   private:
    template<typename>
    friend class RewardDistributionBuilder;

    explicit RewardDistribution(ExactDistribution&& exactMasses)
        : kind(Kind::ExactSparse), exactMasses(std::move(exactMasses)), lowerRewardBound(0), upperRewardBound(0) {
        // Intentionally left empty.
    }

    RewardDistribution(std::vector<ValueType>&& categoricalMasses, uint64_t lowerRewardBound, uint64_t upperRewardBound)
        : kind(Kind::Categorical), categoricalMasses(std::move(categoricalMasses)), lowerRewardBound(lowerRewardBound), upperRewardBound(upperRewardBound) {
        // Intentionally left empty.
    }

    static ValueType getAtomValue(uint64_t atom, uint64_t lowerRewardBound, uint64_t upperRewardBound, uint64_t atoms) {
        STORM_LOG_ASSERT(atoms > 0, "Expected a positive atom count.");
        if (atoms == 1 || lowerRewardBound == upperRewardBound) {
            return storm::utility::convertNumber<ValueType, uint64_t>(lowerRewardBound);
        }
        ValueType const lower = storm::utility::convertNumber<ValueType, uint64_t>(lowerRewardBound);
        ValueType const width = storm::utility::convertNumber<ValueType, uint64_t>(upperRewardBound - lowerRewardBound);
        ValueType const numerator = storm::utility::convertNumber<ValueType, uint64_t>(atom);
        ValueType const denominator = storm::utility::convertNumber<ValueType, uint64_t>(atoms - 1);
        return lower + width * numerator / denominator;
    }

    Kind kind;
    ExactDistribution exactMasses;
    std::vector<ValueType> categoricalMasses;
    uint64_t lowerRewardBound;
    uint64_t upperRewardBound;
};

template<typename ValueType>
class RewardDistributionBuilder {
   public:
    using Distribution = RewardDistribution<ValueType>;
    using Representation = typename RewardDistributionOptions::Representation;

    explicit RewardDistributionBuilder(RewardDistributionOptions const& options) : lowerRewardBound(0), upperRewardBound(options.getCategoricalUpperRewardBound()) {
        STORM_LOG_THROW(options.representation != Representation::Quantile, storm::exceptions::NotSupportedException,
                        "Quantile reward distributions are not implemented yet.");
        STORM_LOG_THROW(options.representation == Representation::Categorical, storm::exceptions::InvalidArgumentException,
                        "Only categorical reward distributions are currently supported.");
        STORM_LOG_THROW(options.atoms > 0, storm::exceptions::InvalidArgumentException, "Categorical reward distributions require a positive atom count.");
        STORM_LOG_THROW(options.stepSize > 0, storm::exceptions::InvalidArgumentException,
                        "Categorical reward distributions require a positive reward step size.");
        categoricalMasses = std::vector<ValueType>(options.atoms, storm::utility::zero<ValueType>());
    }

    void addScaledShifted(ValueType const& scale, Distribution const& distribution, uint64_t rewardShift) {
        if (storm::utility::isZero(scale)) {
            return;
        }
        distribution.forEachMass([this, &scale, rewardShift](ValueType const& reward, ValueType const& mass) {
            ValueType const scaledMass = scale * mass;
            if (storm::utility::isZero(scaledMass)) {
                return;
            }
            ValueType const shiftedReward = reward + storm::utility::convertNumber<ValueType, uint64_t>(rewardShift);
            addCategoricalMass(shiftedReward, scaledMass);
        });
    }

    Distribution build() && {
        return Distribution(std::move(categoricalMasses), lowerRewardBound, upperRewardBound);
    }

   private:
    void addCategoricalMass(ValueType const& reward, ValueType const& mass) {
        addCategoricalMass(reward, mass, categoricalMasses);
    }

    void addCategoricalMass(ValueType const& reward, ValueType const& mass, std::vector<ValueType>& masses) const {
        STORM_LOG_ASSERT(!masses.empty(), "Expected categorical masses to be initialized.");
        ValueType const lower = storm::utility::convertNumber<ValueType, uint64_t>(lowerRewardBound);
        ValueType const upper = storm::utility::convertNumber<ValueType, uint64_t>(upperRewardBound);
        if (masses.size() == 1 || lowerRewardBound == upperRewardBound || reward <= lower) {
            masses.front() += mass;
            return;
        }
        if (reward >= upper) {
            masses.back() += mass;
            return;
        }

        ValueType const relativePosition = (reward - lower) * storm::utility::convertNumber<ValueType, uint64_t>(masses.size() - 1) / (upper - lower);
        double const relativePositionAsDouble = storm::utility::convertNumber<double, ValueType>(relativePosition);
        uint64_t const leftAtom = std::min<uint64_t>(static_cast<uint64_t>(std::floor(relativePositionAsDouble)), masses.size() - 2);
        ValueType const leftValue = Distribution::getAtomValue(leftAtom, lowerRewardBound, upperRewardBound, masses.size());
        ValueType const rightValue = Distribution::getAtomValue(leftAtom + 1, lowerRewardBound, upperRewardBound, masses.size());
        ValueType const rightWeight = (reward - leftValue) / (rightValue - leftValue);
        masses[leftAtom] += mass * (storm::utility::one<ValueType>() - rightWeight);
        masses[leftAtom + 1] += mass * rightWeight;
    }

    uint64_t lowerRewardBound;
    uint64_t upperRewardBound;
    std::vector<ValueType> categoricalMasses;
};

}  // namespace distributional
}  // namespace modelchecker
}  // namespace storm
