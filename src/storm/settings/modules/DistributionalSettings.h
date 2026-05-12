#pragma once

#include <cstdint>
#include <string>

#include "storm/settings/modules/ModuleSettings.h"

namespace storm {
namespace settings {
namespace modules {

class DistributionalSettings : public ModuleSettings {
   public:
    enum class Representation { Categorical, Quantile };

    DistributionalSettings();

    Representation getRepresentation() const;
    uint64_t getNumberOfAtoms() const;
    double getPrecision() const;
    uint64_t getMaximalIterationCount() const;
    uint64_t getNumberOfBudgetAtoms() const;

    static const std::string moduleName;

   private:
    static const std::string representationOptionName;
    static const std::string atomsOptionName;
    static const std::string precisionOptionName;
    static const std::string maxIterationsOptionName;
    static const std::string budgetAtomsOptionName;
};

}  // namespace modules
}  // namespace settings
}  // namespace storm
