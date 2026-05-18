#include "storm/logic/DistributionalFormula.h"

#include <boost/any.hpp>
#include <ostream>

#include "storm/exceptions/InvalidArgumentException.h"
#include "storm/logic/FormulaVisitor.h"
#include "storm/utility/macros.h"

namespace storm {
namespace logic {

DistributionalFormula::DistributionalFormula(std::shared_ptr<Formula const> subformula) : subformula(subformula) {
    STORM_LOG_THROW(static_cast<bool>(subformula), storm::exceptions::InvalidArgumentException, "Distributional formula without subformula is invalid.");
}

DistributionalFormula::~DistributionalFormula() {
    // Intentionally left empty.
}

bool DistributionalFormula::isDistributionalFormula() const {
    return true;
}

bool DistributionalFormula::hasQualitativeResult() const {
    return false;
}

bool DistributionalFormula::hasQuantitativeResult() const {
    return false;
}

Formula const& DistributionalFormula::getSubformula() const {
    return *subformula;
}

boost::any DistributionalFormula::accept(FormulaVisitor const& visitor, boost::any const& data) const {
    return visitor.visit(*this, data);
}

void DistributionalFormula::gatherAtomicExpressionFormulas(std::vector<std::shared_ptr<AtomicExpressionFormula const>>& atomicExpressionFormulas) const {
    subformula->gatherAtomicExpressionFormulas(atomicExpressionFormulas);
}

void DistributionalFormula::gatherAtomicLabelFormulas(std::vector<std::shared_ptr<AtomicLabelFormula const>>& atomicLabelFormulas) const {
    subformula->gatherAtomicLabelFormulas(atomicLabelFormulas);
}

void DistributionalFormula::gatherReferencedRewardModels(std::set<std::string>& referencedRewardModels) const {
    subformula->gatherReferencedRewardModels(referencedRewardModels);
}

void DistributionalFormula::gatherUsedVariables(std::set<storm::expressions::Variable>& usedVariables) const {
    subformula->gatherUsedVariables(usedVariables);
}

std::ostream& DistributionalFormula::writeToStream(std::ostream& out, bool /* allowParentheses */) const {
    out << "distributional(";
    subformula->writeToStream(out);
    out << ")";
    return out;
}

}  // namespace logic
}  // namespace storm
