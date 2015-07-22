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
package learning.supervised.evaluation.roc;

import data.representation.util.DataMineConstants;
import java.io.PrintWriter;
import java.util.ArrayList;
import util.AuxSort;
import util.SOPLUtil;

/**
 * Object of this class represent an ROC curve for a single classifier.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ROCObject {

    private ArrayList<Float> truePositiveVals;
    private ArrayList<Float> falsePositiveVals;
    private float areaUnderCurve = 0;

    /**
     * @param pw PrintWriter object to print the values to the stream.
     */
    public void printToStream(PrintWriter pw) {
        pw.println();
        pw.println("False positive vals: ");
        SOPLUtil.printArrayListToStream(falsePositiveVals, pw);
        pw.println();
        pw.println("True positive vals: ");
        SOPLUtil.printArrayListToStream(truePositiveVals, pw);
        pw.println();
        pw.println("AUC: " + areaUnderCurve);
    }

    /**
     * The default constructor.
     */
    public ROCObject() {
        truePositiveVals = new ArrayList<>(100);
        falsePositiveVals = new ArrayList<>(100);
    }

    /**
     * Initialization and AUC calculation.
     *
     * @param classIndex Integer that is the class index.
     * @param labels int[] representing the data labels.
     * @param classificationProbabilities float[][] representing the class
     * affiliation probabilities for all the points.
     * @param classNegative Boolean flag indicating whether the class index
     * denotes the positive or the negative class. All other classes are merged
     * into its complement.
     */
    public ROCObject(int classIndex, int[] labels,
            float[][] classificationProbabilities, boolean classNegative) {
        truePositiveVals = new ArrayList<>(100);
        falsePositiveVals = new ArrayList<>(100);
        float totalPositives = 0;
        float totalNegatives = 0;
        float[] nClassProbs = new float[labels.length];
        for (int i = 0; i < labels.length; i++) {
            if (classNegative) {
                nClassProbs[i] = classificationProbabilities[i][classIndex];
            } else {
                nClassProbs[i] = 1 - classificationProbabilities[i][classIndex];
            }
            if (labels[i] == classIndex) {
                if (classNegative) {
                    totalNegatives++;
                } else {
                    totalPositives++;
                }
            } else {
                if (classNegative) {
                    totalPositives++;
                } else {
                    totalNegatives++;
                }
            }
        }
        try {
            // Ascending sort.
            int[] indexPermutation = AuxSort.sortIndexedValue(nClassProbs,
                    false);
            int index = 0;
            float truePositiveCount = 0;
            float falsePositiveCount = 0;
            while (index < labels.length) {
                if (classNegative) {
                    while (index < labels.length
                            && (labels[indexPermutation[index]] !=
                            classIndex)) {
                        index++;
                        truePositiveCount++;
                    }
                } else {
                    while (index < labels.length
                            && (labels[indexPermutation[index]] ==
                            classIndex)) {
                        index++;
                        truePositiveCount++;
                    }
                }
                falsePositiveVals.add(falsePositiveCount / totalNegatives);
                truePositiveVals.add(truePositiveCount / totalPositives);
                if (classNegative) {
                    while (index < labels.length
                            && (labels[indexPermutation[index]] ==
                            classIndex)) {
                        index++;
                        falsePositiveCount++;
                    }
                } else {
                    while (index < labels.length
                            && (labels[indexPermutation[index]] !=
                            classIndex)) {
                        index++;
                        falsePositiveCount++;
                    }
                }
            }
            if (falsePositiveVals.get(falsePositiveVals.size() - 1) - 1
                    < DataMineConstants.EPSILON
                    || truePositiveVals.get(truePositiveVals.size() - 1) - 1
                    < DataMineConstants.EPSILON) {
                // The limit case.
                falsePositiveVals.add(1f);
                truePositiveVals.add(1f);
            }
            int cSize = falsePositiveVals.size();
            areaUnderCurve = 0;
            for (int i = 0; i < cSize - 1; i++) {
                // A trapesoid estimate.
                areaUnderCurve += 0.5f * (falsePositiveVals.get(i + 1)
                        - falsePositiveVals.get(i))
                        * (truePositiveVals.get(i + 1) +
                        truePositiveVals.get(i));
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }
}
