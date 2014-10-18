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
package data.generators;

import data.representation.util.DataMineConstants;
import java.util.Random;

/**
 * The idea for this class was to have a generator capable of generating
 * clusters of points that would not be entirely spherical. In particular, this
 * implementation generates clusters that consist of paired overlapping Gaussian
 * distributions. Additionally, there will be a difference in relative density
 * between the two Gaussian generated volumes, so that the centroid would not be
 * well centered. This makes for an interesting test-shape for clustering.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class PairedSphericGaussians implements DataGenerator {

    // Mean, standard deviation and lower and upper value bounds for all the
    // float features that will be generated.
    private float[] mean;
    private float[] stDev;
    private float[] lowerBounds;
    private float[] upperBounds;
    // Default robability of belonging to the first subcluster.
    private float firstProb = 0.67f;

    /**
     * The two subcluster means will be at mean - stDev/2 and mean + stDev/2.
     * The proportion of elements in the two subclusters is governed by a biased
     * coin toss.
     *
     * @param mean Float array representing float feature means.
     * @param stDev Float array representing float feature standard deviations.
     * @param lowerBounds Float array representing float feature value lower
     * bounds.
     * @param upperBounds Float array representing float feature value upper
     * bounds.
     */
    public PairedSphericGaussians(
            float[] mean,
            float[] stDev,
            float[] lowerBounds,
            float[] upperBounds,
            float firstProb) {
        this.mean = mean;
        this.stDev = stDev;
        this.lowerBounds = lowerBounds;
        this.upperBounds = upperBounds;
        this.firstProb = firstProb;
    }

    @Override
    public float[] generateFloat() {
        Random randa = new Random();
        float decision;
        float[] instance = new float[mean.length];
        firstProb = 0.6f + 0.2f * randa.nextFloat();
        for (int i = 0; i < instance.length; i++) {
            decision = randa.nextFloat();
            if (decision > firstProb) {
                decision = randa.nextFloat();
                if (decision < 0.5) {
                    instance[i] = mean[i] - (stDev[i] / 2f) + stDev[i]
                            * (float) randa.nextGaussian();
                } else {
                    instance[i] = mean[i] - (stDev[i] / 2f) - stDev[i]
                            * (float) randa.nextGaussian();
                }
            } else {
                decision = randa.nextFloat();
                if (decision < 0.5) {
                    instance[i] = mean[i] + (stDev[i] / 2f) + stDev[i]
                            * (float) randa.nextGaussian();
                } else {
                    instance[i] = mean[i] + (stDev[i] / 2f) - stDev[i]
                            * (float) randa.nextGaussian();
                }
            }
            // Validation.
            instance[i] = Math.max(lowerBounds[i], instance[i]);
            instance[i] = Math.min(upperBounds[i], instance[i]);
        }
        return instance;
    }

    @Override
    public int[] generateInt() {
        Random randa = new Random();
        float decision;
        int[] instance = new int[mean.length];
        firstProb = 0.6f + 0.2f * randa.nextFloat();
        for (int i = 0; i < instance.length; i++) {
            decision = randa.nextFloat();
            if (decision > firstProb) {
                decision = randa.nextFloat();
                if (decision < 0.5) {
                    instance[i] = (int) (mean[i] - (stDev[i] / 2f)
                            + stDev[i] * (float) randa.nextGaussian());
                } else {
                    instance[i] = (int) (mean[i] - (stDev[i] / 2f)
                            - stDev[i] * (float) randa.nextGaussian());
                }
            } else {
                decision = randa.nextFloat();
                if (decision < 0.5) {
                    instance[i] = (int) (mean[i] + (stDev[i] / 2f)
                            + stDev[i] * (float) randa.nextGaussian());
                } else {
                    instance[i] = (int) (mean[i] + (stDev[i] / 2f)
                            - stDev[i] * (float) randa.nextGaussian());
                }
            }
            // Validation.
            if (instance[i] < lowerBounds[i]) {
                if ((lowerBounds[i] - (int) lowerBounds[i])
                        > DataMineConstants.EPSILON) {
                    instance[i] = (int) (lowerBounds[i]) + 1;
                } else {
                    instance[i] = (int) (lowerBounds[i]);
                }
            }
            if (instance[i] > upperBounds[i]) {
                instance[i] = (int) (upperBounds[i]);
            }
        }
        return instance;
    }
}
