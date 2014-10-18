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
package learning.unsupervised.evaluation;

/**
 * L method is described in "Determining the number of clusters/segments in
 * hierarchical clustering/segmentation algorithms" by Salvador, Chan 2004.
 * Essentially, this method finds a knee in the curve that plots the number of
 * clusters vs the quality index for that number of clusters.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class LMethod {

    private float[] evalGraph;
    private int kneeIndex;

    /**
     * @param evalArray needs to be given so that index 0 refers to eval value
     * at 1 clusters
     */
    public LMethod(float[] evalArray) {
        evalGraph = evalArray;
    }

    /**
     * Finds the best configuration.
     */
    public void findBestConfiguration() {
        kneeIndex = evalGraph.length - 1;
        findKnee(evalGraph.length);
    }

    /**
     * @return The number of clusters.
     */
    public int getNumClust() {
        return kneeIndex + 2;
    }

    /**
     * Find the knee on the evaluation curve.
     *
     * @param lastIndex Last index.
     */
    private void findKnee(int lastIndex) {
        if (lastIndex <= 3) {
            return;
        }
        // First find all possible line fits, then evaluate them.
        double[] leftA, leftB, rightA, rightB;
        leftA = new double[lastIndex - 2];
        leftB = new double[lastIndex - 2];
        rightA = new double[lastIndex - 2];
        rightB = new double[lastIndex - 2];
        double sumY = evalGraph[0];
        double sumX = 0;
        double sumXsq = 0;
        double sumXY = 0;
        for (int i = 1; i <= (lastIndex - 2); i++) {
            sumX += i;
            sumXsq += i * i;
            sumY += evalGraph[i];
            sumXY += i * evalGraph[i];
            leftA[i - 1] = (sumXY - (1 / (i + 1) * sumX * sumY))
                    / (sumXsq - sumX * sumX * (1 / (i + 1)));
            leftB[i - 1] = (1 / (i + 1)) * (sumY - leftA[i] * sumX);
        }
        sumY = evalGraph[lastIndex];
        sumX = 0;
        sumXsq = 0;
        sumXY = 0;
        for (int i = lastIndex - 1; i >= 2; i--) {
            sumX += i;
            sumXsq += i * i;
            sumY += evalGraph[i];
            sumXY += i * evalGraph[i];
            rightA[i - 2] = (sumXY - (1 / (i + 1) * sumX * sumY))
                    / (sumXsq - sumX * sumX * (1 / (i + 1)));
            rightB[i - 2] = (1 / (i + 1)) * (sumY - leftA[i] * sumX);
        }
        double RMSE, LMSE;
        double currError;
        double minError = Float.MAX_VALUE;
        int bestFitIndex = 1;
        double tmp;
        for (int i = 1; i <= lastIndex - 2; i++) {
            LMSE = 0;
            RMSE = 0;
            for (int j = 0; j <= i; j++) {
                tmp = leftA[i - 1] * j + leftB[i - 1];
                LMSE += (tmp - evalGraph[j]) * (tmp - evalGraph[j]);
            }
            LMSE /= (i + 1);
            LMSE = Math.sqrt(LMSE);
            for (int j = i + 1; j <= lastIndex; j++) {
                tmp = rightA[i - 1] * j + rightB[i - 1];
                RMSE += (tmp - evalGraph[j]) * (tmp - evalGraph[j]);
            }
            RMSE /= (lastIndex - i);
            RMSE = Math.sqrt(RMSE);
            currError = (1 / (lastIndex + 1)) * (LMSE * (i + 1) + RMSE
                    * (lastIndex - i));
            if (currError < minError) {
                minError = currError;
                bestFitIndex = i - 1;
            }
        }
        // So, the best fit index is found, now find the knee and reiterate.
        double dKnee = ((rightB[bestFitIndex] - leftB[bestFitIndex])
                / (leftA[bestFitIndex] - rightA[bestFitIndex]));
        int currKnee = (int) dKnee;
        if (dKnee - currKnee > 0.5) {
            currKnee++;
        }
        int tempIndex;
        if (currKnee < kneeIndex) {
            tempIndex = kneeIndex;
            kneeIndex = currKnee;
            findKnee(Math.min(tempIndex * 2, lastIndex - 1));
        }
    }
}
