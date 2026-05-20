#include "storm-config.h"
#include "test/storm_gtest.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "storm-parsers/parser/FormulaParser.h"
#include "storm/adapters/RationalNumberAdapter.h"
#include "storm/environment/Environment.h"
#include "storm/exceptions/NotSupportedException.h"
#include "storm/logic/DistributionalFormula.h"
#include "storm/modelchecker/CheckTask.h"
#include "storm/modelchecker/distributional/DistributionalReachabilityPreprocessor.h"
#include "storm/modelchecker/distributional/DistributionalRewardReachabilityQuery.h"
#include "storm/modelchecker/prctl/SparseMdpPrctlModelChecker.h"
#include "storm/modelchecker/results/ExplicitDistributionalCheckResult.h"
#include "storm/models/sparse/Mdp.h"
#include "storm/models/sparse/StandardRewardModel.h"
#include "storm/models/sparse/StateLabeling.h"
#include "storm/storage/BitVector.h"
#include "storm/storage/SparseMatrix.h"

namespace {

using Mdp = storm::models::sparse::Mdp<double>;
using RewardModel = storm::models::sparse::StandardRewardModel<double>;

storm::storage::SparseMatrix<double> buildTwoStateTargetMatrix() {
    storm::storage::SparseMatrixBuilder<double> builder(3, 2, 3, true, true, 2);
    builder.newRowGroup(0);
    builder.addNextValue(0, 1, 1.0);
    builder.addNextValue(1, 1, 1.0);
    builder.newRowGroup(2);
    builder.addNextValue(2, 0, 1.0);
    return builder.build();
}

storm::storage::SparseMatrix<double> buildTwoStateTransitionRewardMatrix() {
    return buildTwoStateTargetMatrix();
}

storm::models::sparse::StateLabeling makeLabeling(uint64_t stateCount, std::vector<uint64_t> const& initialStates, std::vector<uint64_t> const& targetStates) {
    storm::models::sparse::StateLabeling labeling(stateCount);
    labeling.addLabel("init");
    for (auto state : initialStates) {
        labeling.addLabelToState("init", state);
    }
    labeling.addLabel("target");
    for (auto state : targetStates) {
        labeling.addLabelToState("target", state);
    }
    return labeling;
}

Mdp makeMdp(storm::storage::SparseMatrix<double>&& transitionMatrix, storm::models::sparse::StateLabeling&& labeling, RewardModel&& rewardModel) {
    std::unordered_map<std::string, RewardModel> rewardModels;
    rewardModels.emplace("", std::move(rewardModel));
    return Mdp(std::move(transitionMatrix), std::move(labeling), std::move(rewardModels));
}

Mdp makeTwoStateTargetMdp(std::vector<double> stateRewards, std::vector<double> stateActionRewards, std::vector<uint64_t> initialStates = {0},
                          std::optional<storm::storage::SparseMatrix<double>> transitionRewards = std::nullopt) {
    RewardModel rewardModel(std::optional<std::vector<double>>(std::move(stateRewards)), std::optional<std::vector<double>>(std::move(stateActionRewards)),
                            std::move(transitionRewards));
    return makeMdp(buildTwoStateTargetMatrix(), makeLabeling(2, initialStates, {1}), std::move(rewardModel));
}

Mdp makeImproperInitialMdp() {
    storm::storage::SparseMatrixBuilder<double> builder(2, 2, 2, true, true, 2);
    builder.newRowGroup(0);
    builder.addNextValue(0, 0, 1.0);
    builder.newRowGroup(1);
    builder.addNextValue(1, 1, 1.0);
    RewardModel rewardModel(std::optional<std::vector<double>>(std::vector<double>{0.0, 0.0}),
                            std::optional<std::vector<double>>(std::vector<double>{0.0, 0.0}));
    return makeMdp(builder.build(), makeLabeling(2, {0}, {1}), std::move(rewardModel));
}

std::shared_ptr<storm::logic::DistributionalFormula const> makeDistributionalFormula(std::string const& formulaText) {
    storm::parser::FormulaParser formulaParser;
    return std::make_shared<storm::logic::DistributionalFormula>(formulaParser.parseSingleFormulaFromString(formulaText));
}

storm::modelchecker::distributional::DistributionalReachabilityPreprocessor<Mdp>::Result preprocess(Mdp const& mdp,
                                                                                                    std::string const& formulaText = "Rmin=? [F \"target\"]") {
    storm::Environment env;
    auto formula = makeDistributionalFormula(formulaText);
    auto query = storm::modelchecker::distributional::parseDistributionalRewardReachabilityQuery(*formula);
    return storm::modelchecker::distributional::DistributionalReachabilityPreprocessor<Mdp>::preprocess(env, mdp, query, false);
}

}  // namespace

TEST(SparseMdpDistributionalPreprocessorTest, NormalizesRewardsAndMakesTargetsAbsorbing) {
    auto mdp = makeTwoStateTargetMdp({10.0, 20.0}, {1.0, 2.0, 3.0});

    auto result = preprocess(mdp);

    ASSERT_EQ(3ul, result.stateActionRewards.size());
    EXPECT_DOUBLE_EQ(11.0, result.stateActionRewards[0]);
    EXPECT_DOUBLE_EQ(12.0, result.stateActionRewards[1]);
    EXPECT_DOUBLE_EQ(0.0, result.stateActionRewards[2]);
    EXPECT_TRUE(result.targetStates.get(1));
    EXPECT_TRUE(result.properStates.get(0));
    EXPECT_TRUE(result.properStates.get(1));

    auto targetRow = result.targetAbsorbingTransitionMatrix.getRow(2);
    ASSERT_EQ(1ul, targetRow.getNumberOfEntries());
    auto entry = *targetRow.begin();
    EXPECT_EQ(1ul, entry.getColumn());
    EXPECT_DOUBLE_EQ(1.0, entry.getValue());
}

TEST(SparseMdpDistributionalPreprocessorTest, RejectsTransitionRewards) {
    auto mdp =
        makeTwoStateTargetMdp({0.0, 0.0}, {1.0, 0.0, 0.0}, {0}, std::optional<storm::storage::SparseMatrix<double>>(buildTwoStateTransitionRewardMatrix()));

    STORM_SILENT_EXPECT_THROW(preprocess(mdp), storm::exceptions::NotSupportedException);
}

TEST(SparseMdpDistributionalPreprocessorTest, RejectsNegativeAndNonIntegerRewards) {
    auto negativeRewardMdp = makeTwoStateTargetMdp({0.0, 0.0}, {-1.0, 0.0, 0.0});
    STORM_SILENT_EXPECT_THROW(preprocess(negativeRewardMdp), storm::exceptions::NotSupportedException);

    auto nonIntegerRewardMdp = makeTwoStateTargetMdp({0.0, 0.0}, {0.5, 0.0, 0.0});
    STORM_SILENT_EXPECT_THROW(preprocess(nonIntegerRewardMdp), storm::exceptions::NotSupportedException);
}

TEST(SparseMdpDistributionalPreprocessorTest, RejectsMultipleInitialStatesAndImproperInitialState) {
    auto multipleInitialStatesMdp = makeTwoStateTargetMdp({0.0, 0.0}, {0.0, 0.0, 0.0}, {0, 1});
    STORM_SILENT_EXPECT_THROW(preprocess(multipleInitialStatesMdp), storm::exceptions::NotSupportedException);

    auto improperInitialStateMdp = makeImproperInitialMdp();
    STORM_SILENT_EXPECT_THROW(preprocess(improperInitialStateMdp), storm::exceptions::NotSupportedException);
}

TEST(SparseMdpDistributionalModelCheckingTest, ReturnsDistributionalResultThroughSparseMdpPrctlPath) {
    auto mdp = makeTwoStateTargetMdp({0.0, 0.0}, {3.0, 5.0, 0.0});
    storm::modelchecker::SparseMdpPrctlModelChecker<Mdp> checker(mdp);
    storm::Environment env;
    auto formula = makeDistributionalFormula("Rmin=? [F \"target\"]");

    auto result = checker.check(env, *formula);

    ASSERT_TRUE(result->isExplicitDistributionalCheckResult());
    EXPECT_FALSE(result->isQuantitative());
    auto const& distributionalResult = result->asExplicitDistributionalCheckResult<double>();
    ASSERT_TRUE(distributionalResult.hasFiniteDistribution(0));
    EXPECT_DOUBLE_EQ(3.0, distributionalResult.getExpectedValue(0));
}

TEST(SparseMdpDistributionalModelCheckingTest, RejectsUnsupportedOptimizationAndSchedulerProduction) {
    auto mdp = makeTwoStateTargetMdp({0.0, 0.0}, {3.0, 5.0, 0.0});
    storm::modelchecker::SparseMdpPrctlModelChecker<Mdp> checker(mdp);
    storm::Environment env;

    auto maxFormula = makeDistributionalFormula("Rmax=? [F \"target\"]");
    STORM_SILENT_EXPECT_THROW(checker.check(env, *maxFormula), storm::exceptions::NotSupportedException);

    auto schedulerFormula = makeDistributionalFormula("Rmin=? [F \"target\"]");
    storm::modelchecker::CheckTask<storm::logic::Formula, double> schedulerTask(*schedulerFormula);
    schedulerTask.setProduceSchedulers();
    STORM_SILENT_EXPECT_THROW(checker.check(env, schedulerTask), storm::exceptions::NotSupportedException);
}
