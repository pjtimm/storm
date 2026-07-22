#include "storm-config.h"
#include "test/storm_gtest.h"

#include <vector>

#include "storm/exceptions/NotSupportedException.h"
#include "storm/modelchecker/distributional/SparseMdpCvarPreprocessor.h"
#include "storm/storage/BitVector.h"
#include "storm/storage/SparseMatrix.h"

TEST(SparseMdpCvarPreprocessorTest, ComputesBoundsForProperDag) {
    storm::storage::SparseMatrixBuilder<double> builder(5, 4, 5, true, true, 4);
    builder.newRowGroup(0);
    builder.addNextValue(0, 1, 1.0);
    builder.addNextValue(1, 2, 1.0);
    builder.newRowGroup(2);
    builder.addNextValue(2, 3, 1.0);
    builder.newRowGroup(3);
    builder.addNextValue(3, 3, 1.0);
    builder.newRowGroup(4);
    builder.addNextValue(4, 3, 1.0);
    auto matrix = builder.build();

    std::vector<double> rewards = {2.0, 3.0, 4.0, 1.0, 0.0};
    storm::storage::BitVector targetStates(4, std::vector<uint64_t>{3});
    storm::storage::BitVector properStates(4, true);

    storm::modelchecker::distributional::SparseMdpCvarPreprocessor<double> preprocessor(matrix, rewards, targetStates, properStates, 0);
    auto result = preprocessor.computeRewardBounds();

    EXPECT_TRUE(result.finiteRewardStates.get(0));
    EXPECT_EQ(0ul, result.initialState);
    ASSERT_EQ(4ul, result.lowerRewardBounds.size());
    ASSERT_EQ(4ul, result.upperRewardBounds.size());
    EXPECT_DOUBLE_EQ(4.0, result.lowerRewardBounds[0]);
    EXPECT_DOUBLE_EQ(6.0, result.upperRewardBounds[0]);
    EXPECT_DOUBLE_EQ(4.0, result.lowerRewardBounds[1]);
    EXPECT_DOUBLE_EQ(4.0, result.upperRewardBounds[1]);
    EXPECT_DOUBLE_EQ(1.0, result.lowerRewardBounds[2]);
    EXPECT_DOUBLE_EQ(1.0, result.upperRewardBounds[2]);
    EXPECT_DOUBLE_EQ(0.0, result.lowerRewardBounds[3]);
    EXPECT_DOUBLE_EQ(0.0, result.upperRewardBounds[3]);
    EXPECT_DOUBLE_EQ(4.0, result.initialLowerRewardBound);
    EXPECT_DOUBLE_EQ(6.0, result.initialUpperRewardBound);
    ASSERT_EQ(7ul, result.getNumberOfBudgetAtoms());
    EXPECT_EQ(4ul, result.getFirstInitialBudgetIndex());
    EXPECT_DOUBLE_EQ(0.0, result.getBudgetValue(0));
    EXPECT_DOUBLE_EQ(4.0, result.getBudgetValue(4));
    EXPECT_DOUBLE_EQ(6.0, result.getBudgetValue(6));
    EXPECT_EQ(4ul, result.getNextBudgetIndex(6, 2.0));
    EXPECT_EQ(5ul, result.getNextBudgetIndex(6, 1.0));
}

TEST(SparseMdpCvarPreprocessorTest, BuildsExactIntegerBudgetGrid) {
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

    storm::modelchecker::distributional::SparseMdpCvarPreprocessor<double> preprocessor(matrix, rewards, targetStates, properStates, 0);
    auto result = preprocessor.computeRewardBounds();

    ASSERT_EQ(11ul, result.getNumberOfBudgetAtoms());
    for (uint64_t index = 0; index < result.getNumberOfBudgetAtoms(); ++index) {
        EXPECT_DOUBLE_EQ(static_cast<double>(index), result.getBudgetValue(index));
    }
    EXPECT_EQ(6ul, result.getNextBudgetIndex(10, 4.0));
    EXPECT_EQ(2ul, result.getNextBudgetIndex(2, 0.0));
    EXPECT_EQ(0ul, result.getNextBudgetIndex(1, 20.0));
}

TEST(SparseMdpCvarPreprocessorTest, BuildsResidualGridForSingletonInitialSupport) {
    storm::storage::SparseMatrixBuilder<double> builder(2, 2, 2, true, true, 2);
    builder.newRowGroup(0);
    builder.addNextValue(0, 1, 1.0);
    builder.newRowGroup(1);
    builder.addNextValue(1, 1, 1.0);
    auto matrix = builder.build();

    std::vector<double> rewards = {5.0, 0.0};
    storm::storage::BitVector targetStates(2, std::vector<uint64_t>{1});
    storm::storage::BitVector properStates(2, true);

    storm::modelchecker::distributional::SparseMdpCvarPreprocessor<double> preprocessor(matrix, rewards, targetStates, properStates, 0);
    auto result = preprocessor.computeRewardBounds();

    ASSERT_EQ(6ul, result.getNumberOfBudgetAtoms());
    EXPECT_EQ(5ul, result.getFirstInitialBudgetIndex());
    EXPECT_DOUBLE_EQ(0.0, result.getBudgetValue(0));
    EXPECT_DOUBLE_EQ(5.0, result.getBudgetValue(5));
    EXPECT_EQ(5ul, result.getNextBudgetIndex(5, 0.0));
    EXPECT_EQ(0ul, result.getNextBudgetIndex(5, 5.0));
}

TEST(SparseMdpCvarPreprocessorTest, RejectsProperNonTargetCycle) {
    storm::storage::SparseMatrixBuilder<double> builder(4, 3, 4, true, true, 3);
    builder.newRowGroup(0);
    builder.addNextValue(0, 1, 1.0);
    builder.addNextValue(1, 2, 1.0);
    builder.newRowGroup(2);
    builder.addNextValue(2, 0, 1.0);
    builder.newRowGroup(3);
    builder.addNextValue(3, 2, 1.0);
    auto matrix = builder.build();

    std::vector<double> rewards = {1.0, 0.0, 1.0, 0.0};
    storm::storage::BitVector targetStates(3, std::vector<uint64_t>{2});
    storm::storage::BitVector properStates(3, true);

    storm::modelchecker::distributional::SparseMdpCvarPreprocessor<double> preprocessor(matrix, rewards, targetStates, properStates, 0);
    STORM_SILENT_EXPECT_THROW(preprocessor.computeRewardBounds(), storm::exceptions::NotSupportedException);
}

TEST(SparseMdpCvarPreprocessorTest, RejectsImproperStates) {
    storm::storage::SparseMatrixBuilder<double> builder(4, 3, 4, true, true, 3);
    builder.newRowGroup(0);
    builder.addNextValue(0, 1, 1.0);
    builder.addNextValue(1, 2, 1.0);
    builder.newRowGroup(2);
    builder.addNextValue(2, 1, 1.0);
    builder.newRowGroup(3);
    builder.addNextValue(3, 2, 1.0);
    auto matrix = builder.build();

    std::vector<double> rewards = {4.0, 0.0, 0.0, 0.0};
    storm::storage::BitVector targetStates(3, std::vector<uint64_t>{1});
    storm::storage::BitVector properStates(3, std::vector<uint64_t>{0, 1});

    auto constructPreprocessor = [&]() {
        storm::modelchecker::distributional::SparseMdpCvarPreprocessor<double> preprocessor(matrix, rewards, targetStates, properStates, 0);
        static_cast<void>(preprocessor);
    };
    STORM_SILENT_EXPECT_THROW(constructPreprocessor(), storm::exceptions::NotSupportedException);
}
