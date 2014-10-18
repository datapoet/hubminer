/**
* Hub Miner: a hubness-aware machine learning experimentation library.
* Copyright (C) 2014  Nenad Tomasev. Email: nenad.tomasev at gmail.com
* 
* This program is free software: you can redistribute it and/or modify it under
* the terms of the GNU General Public License as published by the Free Software
* Foundation, either version 3 of the License, or (at your option) any later
* version.
* 
* This program is distributed in the hope that it will be useful, but WITHOUT
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
* FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with
* this program. If not, see <http://www.gnu.org/licenses/>.
*/
package learning.supervised.evaluation;

import data.representation.util.DataMineConstants;
import ioformat.FileUtil;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;

/**
 * This class implements the methods for classification prediction quality
 * estimation in terms of accuracy, precision, recall, f1-score and the Matthews
 * correlation coefficient. It calculates all these measures based on the
 * confusion matrix. In the confusion matrix, the element i,j denotes the
 * (average) number of points that were classified as i despite being of class
 * j.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ClassificationEstimator {

    private float[][] confusionMatrix = null;
    private float accuracy = 0f;
    private float[] precision = null;
    private float[] recall = null;
    private float avgPrecision = 0f;
    private float avgRecall = 0f;
    // Micro-averaged.
    private float fMeasureMicroAveraged = 0f;
    // Macro-averaged.
    private float fMeasureMacroAveraged = 0f;
    // The Matthews correlation coefficient is used in binary classification
    // tasks.
    private float matthewsCorrelation = 0f;

    /**
     * Initialization.
     *
     * @param confusionMatrix float[][] that is the confusion matrix.
     */
    public ClassificationEstimator(float[][] confusionMatrix) {
        this.confusionMatrix = confusionMatrix;
    }

    /**
     * @param matthewsCorrelation Float value that is the Matthews correlation
     * coefficient.
     */
    public void setMatthewsCorrCoef(float matthewsCorrelation) {
        this.matthewsCorrelation = matthewsCorrelation;
    }

    /**
     * @return Float value that is the Matthews correlation coefficient.
     */
    public float getMatthewsCorrCoef() {
        return matthewsCorrelation;
    }

    /**
     * @param fMeasureMacroAveraged Float value that is the macro-averaged F1
     * score.
     */
    public void setMacroFMeasure(float fMeasureMacroAveraged) {
        this.fMeasureMacroAveraged = fMeasureMacroAveraged;
    }

    /**
     * @return Float value that is the macro-averaged F1 score.
     */
    public float getMacroFMeasure() {
        return fMeasureMacroAveraged;
    }

    /**
     * @return float[][] that is the confusion matrix representing the overall
     * class predictions.
     */
    public float[][] getConfusionMatrix() {
        return confusionMatrix;
    }

    /**
     * @param confusionMatrix float[][] that is the confusion matrix
     * representing the overall class predictions.
     */
    public void setConfusionMatrix(float[][] confusionMatrix) {
        this.confusionMatrix = confusionMatrix;
    }

    /**
     * @return Float value that is the average classification accuracy.
     */
    public float getAccuracy() {
        return accuracy;
    }

    /**
     * @param accuracy Float value that is the average classification accuracy.
     */
    public void setAccuracy(float accuracy) {
        this.accuracy = accuracy;
    }

    /**
     * @return float[] that is the array of class-wise precisions.
     */
    public float[] getPrecision() {
        return precision;
    }

    /**
     * @param precision float[] that is the array of class-wise precisions.
     */
    public void setPrecision(float[] precision) {
        this.precision = precision;
    }

    /**
     * @return float[] that is the array of class-wise recalls.
     */
    public float[] getRecall() {
        return recall;
    }

    /**
     * @param recall float[] that is the array of class-wise recalls.
     */
    public void setRecall(float[] recall) {
        this.recall = recall;
    }

    /**
     * @return Float value that is the micro-averaged F1-score.
     */
    public float getMicroFMeasure() {
        return fMeasureMicroAveraged;
    }

    /**
     * @param fMeasure Float value that is the micro-averaged F1-score.
     */
    public void setMicroFMeasure(float fMeasure) {
        this.fMeasureMicroAveraged = fMeasure;
    }

    /**
     * @return Float value that is the macro-averaged precision.
     */
    public float getAvgPrecision() {
        return avgPrecision;
    }

    /**
     * @param avgPrecision Float value that is the macro-averaged precision.
     */
    public void setAvgPrecision(float avgPrecision) {
        this.avgPrecision = avgPrecision;
    }

    /**
     * @return Float value that is the macro-averaged recall.
     */
    public float getAvgRecall() {
        return avgRecall;
    }

    /**
     * @param avgRecall Float value that is the macro-averaged recall.
     */
    public void setAvgRecall(float avgRecall) {
        this.avgRecall = avgRecall;
    }

    /**
     * @param beta Float that is the F-score parameter value.
     * @return Float value that is the F-beta score.
     */
    public float getFMeasure(float beta) {
        return ((1 + beta * beta) * avgPrecision * avgRecall
                / (beta * beta * avgPrecision + avgRecall));
    }

    /**
     * @return True if the underlying classification problem was binary, false
     * otherwise.
     */
    public boolean isBinary() {
        if (confusionMatrix != null) {
            return confusionMatrix.length == 2;
        } else {
            return false;
        }
    }

    /**
     * This method prints the stDevs of the main stats in the array of
     * classification estimators to a file, which is usually done when
     * persisting the average results of cross-validation.
     *
     * @param estimators ClassificationEstimator[] array of classification
     * estimator objects to persist the average main stats StDevs of.
     * @param outFile File to persist the data to.
     * @throws Exception
     */
    public static void appendMainPointsStDevsToFile(
            ClassificationEstimator[] estimators, File outFile)
            throws Exception {
        try (PrintWriter pw =
                new PrintWriter(new FileWriter(outFile, true));) {
            pw.println("accuracyStDev,PrecisionStDev,RecallStDev,"
                    + "fMeasureStDev");
            float accStDev = 0;
            float precStDev = 0;
            float recStDev = 0;
            float fMStDev = 0;
            float accAvg = 0;
            float precAvg = 0;
            float recAvg = 0;
            float fMAvg = 0;
            int numValidEstimators = 0;
            for (int estimatorIndex = 0; estimatorIndex < estimators.length;
                    estimatorIndex++) {
                if (estimators[estimatorIndex] != null) {
                    // Skip invalid estimators.
                    numValidEstimators++;
                    accAvg += estimators[estimatorIndex].accuracy;
                    precAvg += estimators[estimatorIndex].avgPrecision;
                    recAvg += estimators[estimatorIndex].avgRecall;
                    fMAvg += estimators[estimatorIndex].fMeasureMicroAveraged;
                }
            }
            if (numValidEstimators > 0) {
                accAvg /= numValidEstimators;
                precAvg /= numValidEstimators;
                recAvg /= numValidEstimators;
                fMAvg /= numValidEstimators;
                for (int i = 0; i < estimators.length; i++) {
                    if (estimators[i] != null) {
                        numValidEstimators++;
                        accStDev +=
                                Math.pow(accAvg - estimators[i].accuracy, 2);
                        precAvg += Math.pow(
                                precAvg - estimators[i].avgPrecision, 2);
                        recAvg += Math.pow(recAvg - estimators[i].avgRecall, 2);
                        fMAvg += Math.pow(fMAvg
                                - estimators[i].fMeasureMicroAveraged, 2);
                    }
                }
                accStDev /= numValidEstimators;
                precStDev /= numValidEstimators;
                recStDev /= numValidEstimators;
                fMStDev /= numValidEstimators;
                accStDev = (float) Math.sqrt(accStDev);
                precStDev = (float) Math.sqrt(accStDev);
                recStDev = (float) Math.sqrt(accStDev);
                fMStDev = (float) Math.sqrt(accStDev);
            }
            // The averages are printed on the last line.
            pw.println(accStDev + "," + precStDev + "," + recStDev + ","
                    + fMStDev);
        } catch (Exception e) {
            throw e;
        }
    }

    /**
     * This method persists the main stats of an array of estimators to a file.
     *
     * @param estimators
     * @param outFile
     * @throws Exception
     */
    public static void printMainPointsOfEstimatorsToFile(
            ClassificationEstimator[] estimators, File outFile)
            throws Exception {
        FileUtil.createFile(outFile);
        try (PrintWriter pw = new PrintWriter(new FileWriter(outFile));) {
            pw.println("accuracy,avgPrecision,avgRecall,fMeasure,mcc");
            for (int i = 0; i < estimators.length; i++) {
                if (estimators[i] != null) {
                    pw.println(estimators[i].accuracy + ","
                            + estimators[i].avgPrecision + ","
                            + estimators[i].avgRecall + ","
                            + estimators[i].fMeasureMicroAveraged + ","
                            + estimators[i].matthewsCorrelation);
                }
            }
        } catch (Exception e) {
            throw e;
        }
    }

    /**
     * This method prints the estimator classification quality measures to a
     * file.
     *
     * @param outFile File to print the estimator to.
     * @throws Exception
     */
    public void printEstimatorToFile(File outFile) throws Exception {
        FileUtil.createFile(outFile);
        try (PrintWriter pw = new PrintWriter(new FileWriter(outFile));) {
            pw.println("accuracy,avgPrecision,avgRecall,fMeasureMicro,"
                    + "fMeasureMacro,MCC");
            pw.println(accuracy + "," + avgPrecision + "," + avgRecall + ","
                    + fMeasureMicroAveraged + "," + fMeasureMacroAveraged + ","
                    + matthewsCorrelation);
            pw.println();
            pw.println("numClasses: " + precision.length);
            pw.println();
            pw.print("precision array:");
            for (int i = 0; i < precision.length; i++) {
                pw.print(" " + precision[i]);
            }
            pw.println();
            pw.print("recall array:");
            for (int i = 0; i < recall.length; i++) {
                pw.print(" " + recall[i]);
            }
            pw.println();
            pw.println();
            pw.println("confusion matrix");
            for (int i = 0; i < confusionMatrix.length; i++) {
                pw.print(confusionMatrix[i][0]);
                for (int j = 1; j < confusionMatrix[i].length; j++) {
                    pw.print(" " + confusionMatrix[i][j]);
                }
                pw.println();
            }
            pw.println();
        } catch (Exception e) {
            throw e;
        }
    }

    /**
     * This method calculates the classification prediction quality measures
     * based on the confusion matrix.
     */
    public void calculateEstimates() {
        int[] sumRow = new int[confusionMatrix.length];
        int[] sumColumn = new int[confusionMatrix.length];
        precision = new float[confusionMatrix.length];
        float[] fMeasureArr = new float[confusionMatrix.length];
        recall = new float[confusionMatrix.length];
        if (confusionMatrix != null) {
            if (isBinary()) {
                // Calculate the Matthews correlation coefficient.
                matthewsCorrelation = confusionMatrix[0][0]
                        * confusionMatrix[1][1] - confusionMatrix[0][1]
                        * confusionMatrix[1][0];
                float denominator = (float) Math.sqrt((confusionMatrix[0][0]
                        + confusionMatrix[0][1]) * (confusionMatrix[0][0]
                        + confusionMatrix[1][0]) * (confusionMatrix[1][1]
                        + confusionMatrix[0][1]) * (confusionMatrix[1][1]
                        + confusionMatrix[1][0]));
                matthewsCorrelation = DataMineConstants.isPositive(denominator)
                        ? matthewsCorrelation / denominator : 0;
            }
            for (int i = 0; i < confusionMatrix.length; i++) {
                precision[i] = confusionMatrix[i][i];
                recall[i] = precision[i];
                for (int j = 0; j < confusionMatrix.length; j++) {
                    sumRow[i] += confusionMatrix[i][j];
                    sumColumn[j] += confusionMatrix[i][j];
                }
            }
            int numElements = 0;
            for (int i = 0; i < sumRow.length; i++) {
                if (sumRow[i] != 0) {
                    precision[i] /= (float) sumRow[i];
                }
                numElements += sumRow[i];
            }
            for (int i = 0; i < sumColumn.length; i++) {
                if (sumColumn[i] != 0) {
                    recall[i] = recall[i] / (float) sumColumn[i];
                }

            }
            accuracy = 0f;
            for (int i = 0; i < recall.length; i++) {
                if (sumColumn[i] != 0) {
                    accuracy += recall[i] * sumColumn[i];
                }
            }
            accuracy = (float) accuracy / (float) numElements;
        }
        float[] classDistr = new float[sumColumn.length];
        float sizeTotal = 0;
        for (int j = 0; j < sumColumn.length; j++) {
            sizeTotal += sumColumn[j];
        }
        for (int j = 0; j < sumColumn.length; j++) {
            classDistr[j] = (float) sumColumn[j] / sizeTotal;
        }
        fMeasureMacroAveraged = 0;
        fMeasureMicroAveraged = 0;
        for (int i = 0; i < precision.length; i++) {
            avgPrecision += precision[i];
            avgRecall += recall[i];
            if (precision[i] + recall[i] != 0) {
                fMeasureArr[i] = 2 * precision[i] * recall[i]
                        / (precision[i] + recall[i]);
            }
            fMeasureMacroAveraged += fMeasureArr[i];
            fMeasureMicroAveraged += classDistr[i] * fMeasureArr[i];
        }
        avgPrecision /= (float) precision.length;
        avgRecall /= (float) recall.length;
        fMeasureMacroAveraged /= (float) fMeasureArr.length;
    }
}