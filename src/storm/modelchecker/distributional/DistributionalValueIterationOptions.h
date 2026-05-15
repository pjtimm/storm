#pragma once

#include <cstdint>

#include "storm/exceptions/NotSupportedException.h"
#include "storm/modelchecker/distributional/RewardDistribution.h"
#include "storm/settings/modules/DistributionalSettings.h"
#include "storm/utility/macros.h"

namespace storm {
namespace modelchecker {
namespace distributional {

struct DistributionalValueIterationOptions {
    using Representation = storm::settings::modules::DistributionalSettings::Representation;

    Representation representation;
    uint64_t atoms;
    double precision;
    uint64_t maximalIterations;

    RewardDistributionOptions toRewardDistributionOptions() const {
        return RewardDistributionOptions{representation, atoms};
    }

    static DistributionalValueIterationOptions fromSettings(storm::settings::modules::DistributionalSettings const& settings) {
        DistributionalValueIterationOptions options{settings.getRepresentation(), settings.getNumberOfAtoms(), settings.getPrecision(),
                                                    settings.getMaximalIterationCount()};
        options.validate();
        return options;
    }

    void validate() const {
        STORM_LOG_THROW(representation != Representation::Quantile, storm::exceptions::NotSupportedException,
                        "Distributional value iteration does not support quantile reward distributions yet.");
    }
};

}  // namespace distributional
}  // namespace modelchecker
}  // namespace storm
