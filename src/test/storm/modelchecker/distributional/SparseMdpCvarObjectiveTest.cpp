#include "storm-config.h"
#include "test/storm_gtest.h"

#include <vector>

#include "storm/modelchecker/distributional/DistributionalValueIterationOptions.h"
#include "storm/modelchecker/distributional/SparseMdpCvarObjective.h"
#include "storm/modelchecker/distributional/SparseMdpCvarPreprocessor.h"
#include "storm/solver/OptimizationDirection.h"
#include "storm/storage/BitVector.h"
#include "storm/storage/SparseMatrix.h"

namespace {

storm::modelchecker::distributional::DistributionalValueIterationOptions makeCvarOptions(uint64_t atoms = 128, uint64_t stepSize = 1,
                                                                                         uint64_t budgetAtoms = 3) {
    storm::modelchecker::distributional::DistributionalValueIterationOptions options{
        storm::modelchecker::distributional::RewardDistributionRepresentation::Categorical, atoms, stepSize, 1e-10, 10000};
    options.objective = storm::modelchecker::distributional::DistributionalValueIterationOptions::Objective::Cvar;
    options.alpha = 0.5;
    options.budgetAtoms = budgetAtoms;
    return options;
}

double computeBranchingCvarSelectedExpectedValue(storm::solver::OptimizationDirection optimizationDirection,
                                                 storm::modelchecker::distributional::DistributionalCvarInterpretation interpretation) {
    storm::storage::SparseMatrixBuilder<double> builder(8, 7, 10, true, true, 7);
    builder.newRowGroup(0);
    builder.addNextValue(0, 1, 0.85);
    builder.addNextValue(0, 2, 0.15);
    builder.addNextValue(1, 3, 1.0);
    builder.newRowGroup(2);
    builder.addNextValue(2, 6, 1.0);
    builder.newRowGroup(3);
    builder.addNextValue(3, 6, 1.0);
    builder.newRowGroup(4);
    builder.addNextValue(4, 4, 1.0);
    builder.newRowGroup(5);
    builder.addNextValue(5, 5, 0.5);
    builder.addNextValue(5, 6, 0.5);
    builder.newRowGroup(6);
    builder.addNextValue(6, 6, 1.0);
    builder.newRowGroup(7);
    builder.addNextValue(7, 6, 1.0);
    auto matrix = builder.build();

    std::vector<double> rewards = {0.0, 0.0, 2.0, 30.0, 3.0, 3.0, 1.0, 0.0};
    storm::storage::BitVector targetStates(7, std::vector<uint64_t>{6});
    storm::storage::BitVector properStates(7, true);
    auto options = makeCvarOptions(41, 1, 29);
    options.alpha = 0.25;
    options.optimizationDirection = optimizationDirection;
    options.cvarInterpretation = interpretation;

    storm::modelchecker::distributional::SparseMdpCvarPreprocessor<double> preprocessor(matrix, rewards, targetStates, properStates, 0, options.budgetAtoms);
    auto preprocessorResult = preprocessor.computeRewardBounds();
    storm::modelchecker::distributional::SparseMdpCvarObjective<double> objective(matrix, rewards, targetStates, properStates, options, preprocessorResult);
    auto result = objective.computeCvarOptimalDistribution();
    return result.distributions.at(0).getProjectedExpectedValue();
}

}  // namespace

TEST(SparseMdpCvarObjectiveTest, ProductIndexRoundTrips) {
    storm::storage::SparseMatrixBuilder<double> builder(3, 2, 3, true, true, 2);
    builder.newRowGroup(0);
    builder.addNextValue(0, 1, 1.0);
    builder.addNextValue(1, 1, 1.0);
    builder.newRowGroup(2);
    builder.addNextValue(2, 1, 1.0);
    auto matrix = builder.build();

    std::vector<double> rewards = {0.0, 10.0, 0.0};
    storm::storage::BitVector targetStates(2, std::vector<uint64_t>{1});
    storm::storage::BitVector properStates(2, true);
    auto options = makeCvarOptions(128, 1, 4);

    storm::modelchecker::distributional::SparseMdpCvarPreprocessor<double> preprocessor(matrix, rewards, targetStates, properStates, 0, options.budgetAtoms);
    auto preprocessorResult = preprocessor.computeRewardBounds();
    EXPECT_DOUBLE_EQ(0.0, preprocessorResult.initialLowerRewardBound);
    EXPECT_DOUBLE_EQ(10.0, preprocessorResult.initialUpperRewardBound);
    storm::modelchecker::distributional::SparseMdpCvarObjective<double> objective(matrix, rewards, targetStates, properStates, options, preprocessorResult);

    EXPECT_EQ(2ul, objective.getStateCount());
    EXPECT_EQ(4ul, objective.getBudgetCount());
    EXPECT_EQ(8ul, objective.getProductStateCount());
    for (uint64_t state = 0; state < objective.getStateCount(); ++state) {
        for (uint64_t budgetIndex = 0; budgetIndex < objective.getBudgetCount(); ++budgetIndex) {
            uint64_t const productState = objective.getProductStateIndex(state, budgetIndex);
            EXPECT_EQ(state, objective.getOriginalState(productState));
            EXPECT_EQ(budgetIndex, objective.getBudgetIndex(productState));
        }
    }
}

TEST(SparseMdpCvarObjectiveTest, ReturnsOnlyInitialStateDistributionFromTopologicalSweep) {
    storm::storage::SparseMatrixBuilder<double> builder(3, 2, 3, true, true, 2);
    builder.newRowGroup(0);
    builder.addNextValue(0, 1, 1.0);
    builder.addNextValue(1, 1, 1.0);
    builder.newRowGroup(2);
    builder.addNextValue(2, 1, 1.0);
    auto matrix = builder.build();

    std::vector<double> rewards = {0.0, 10.0, 0.0};
    storm::storage::BitVector targetStates(2, std::vector<uint64_t>{1});
    storm::storage::BitVector properStates(2, true);
    auto options = makeCvarOptions(21, 1, 4);
    options.maximalIterations = 0;

    storm::modelchecker::distributional::SparseMdpCvarPreprocessor<double> preprocessor(matrix, rewards, targetStates, properStates, 0, options.budgetAtoms);
    auto preprocessorResult = preprocessor.computeRewardBounds();
    storm::modelchecker::distributional::SparseMdpCvarObjective<double> objective(matrix, rewards, targetStates, properStates, options, preprocessorResult);
    auto result = objective.computeCvarOptimalDistribution();

    ASSERT_EQ(1ul, result.distributions.size());
    ASSERT_EQ(1ul, result.distributions.count(0));
    EXPECT_DOUBLE_EQ(0.0, result.distributions.at(0).getProjectedExpectedValue());
}

TEST(SparseMdpCvarObjectiveTest, AlphaAffectsSelectedInitialBudget) {
    storm::storage::SparseMatrixBuilder<double> builder(4, 3, 5, true, true, 3);
    builder.newRowGroup(0);
    builder.addNextValue(0, 2, 0.9);
    builder.addNextValue(0, 1, 0.1);
    builder.addNextValue(1, 2, 1.0);
    builder.newRowGroup(2);
    builder.addNextValue(2, 2, 1.0);
    builder.newRowGroup(3);
    builder.addNextValue(3, 2, 1.0);
    auto matrix = builder.build();

    std::vector<double> rewards = {0.0, 4.0, 20.0, 0.0};
    storm::storage::BitVector targetStates(3, std::vector<uint64_t>{2});
    storm::storage::BitVector properStates(3, true);
    auto options = makeCvarOptions(21, 1, 5);

    storm::modelchecker::distributional::SparseMdpCvarPreprocessor<double> preprocessor(matrix, rewards, targetStates, properStates, 0, options.budgetAtoms);
    auto preprocessorResult = preprocessor.computeRewardBounds();
    EXPECT_DOUBLE_EQ(0.0, preprocessorResult.initialLowerRewardBound);
    EXPECT_DOUBLE_EQ(20.0, preprocessorResult.initialUpperRewardBound);

    options.alpha = 0.25;
    storm::modelchecker::distributional::SparseMdpCvarObjective<double> smallTailObjective(matrix, rewards, targetStates, properStates, options,
                                                                                           preprocessorResult);
    auto smallTailResult = smallTailObjective.computeCvarOptimalDistribution();

    options.alpha = 0.75;
    storm::modelchecker::distributional::SparseMdpCvarObjective<double> largeTailObjective(matrix, rewards, targetStates, properStates, options,
                                                                                           preprocessorResult);
    auto largeTailResult = largeTailObjective.computeCvarOptimalDistribution();

    EXPECT_DOUBLE_EQ(4.0, smallTailResult.distributions.at(0).getProjectedExpectedValue());
    EXPECT_NEAR(2.0, largeTailResult.distributions.at(0).getProjectedExpectedValue(), 1e-12);
}

TEST(SparseMdpCvarObjectiveTest, SelectsDistributionAccordingToDirectionAndInterpretation) {
    using storm::modelchecker::distributional::DistributionalCvarInterpretation;
    using storm::solver::OptimizationDirection;

    EXPECT_NEAR(6.5, computeBranchingCvarSelectedExpectedValue(OptimizationDirection::Minimize, DistributionalCvarInterpretation::Cost), 1e-12);
    EXPECT_NEAR(6.2, computeBranchingCvarSelectedExpectedValue(OptimizationDirection::Maximize, DistributionalCvarInterpretation::Cost), 1e-12);
    EXPECT_NEAR(6.5, computeBranchingCvarSelectedExpectedValue(OptimizationDirection::Maximize, DistributionalCvarInterpretation::Reward), 1e-12);
    EXPECT_NEAR(6.2, computeBranchingCvarSelectedExpectedValue(OptimizationDirection::Minimize, DistributionalCvarInterpretation::Reward), 1e-12);
}
