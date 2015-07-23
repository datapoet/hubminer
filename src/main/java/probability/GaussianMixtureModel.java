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
package probability;

import learning.unsupervised.Cluster;
import data.representation.DataInstance;
import util.ArrayUtil;
import java.util.ArrayList;

/**
 * Class that implements a simple GaussianMixture model as an array of Gaussian
 * distributions.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class GaussianMixtureModel extends ProbabilityModel {

    GaussianModel[] models;

    /**
     * @param models An array of Gaussian models composing the mixture.
     */
    public GaussianMixtureModel(GaussianModel[] models) {
        this.models = models;
        for (int i = 0; i < models.length; i++) {
            if (models[i].exOp == null) {
                // If the model is not already calculated, calculate it now.
                models[i].calculatePredictiveModel();
            }
        }
    }

    /**
     * Learn a GMM from a set of clusters where each cluster corresponds to a
     * single Gaussian distribution.
     *
     * @param samples An array of clusters.
     */
    public GaussianMixtureModel(Cluster[] samples) {
        ArrayList<Cluster> nonEmpty = new ArrayList<>(samples.length);
        for (Cluster clust : samples) {
            if (clust != null && !clust.isEmpty() && clust.size() > 1) {
                nonEmpty.add(clust);
            }
        }
        Cluster[] sampClean = new Cluster[nonEmpty.size()];
        for (int i = 0; i < sampClean.length; i++) {
            sampClean[i] = nonEmpty.get(i);
        }
        this.models = new GaussianModel[sampClean.length];
        for (int i = 0; i < models.length; i++) {
            models[i] = new GaussianModel(sampClean[i]);
            if (models[i].exOp == null) {
                models[i].calculatePredictiveModel();
            }
        }
    }

    /**
     * Calculate how likely it is for a given float feature vector to have
     * originated from any of the modeled Gaussians.
     *
     * @param x A float array representing a feature vector.
     * @return An array of likelihoods of the feature vector originating from
     * the modeled Gaussian distributions.
     */
    public double[] calcProbArray(float[] x) {
        double[] probs = new double[models.length];
        for (int i = 0; i < probs.length; i++) {
            probs[i] = models[i].getProbability(x);
        }
        return probs;
    }

    /**
     * Calculate how likely it is for a given float feature vector to have
     * originated from any of the modeled Gaussians.
     *
     * @param instance DataInstance object.
     * @return An array of likelihoods of the data instance originating from the
     * modeled Gaussian distributions.
     */
    public double[] calcProbArray(DataInstance instance) {
        double[] probs = new double[models.length];
        for (int i = 0; i < probs.length; i++) {
            probs[i] = models[i].getProbability(instance.fAttr);
        }
        return probs;
    }

    @Override
    public double[] calcTestDataProbabilities(DataInstance[] testArray) {
        double[] probs = new double[testArray.length];
        double[] cProbs;
        int N = testArray.length;
        for (int i = 0; i < N; i++) {
            cProbs = calcProbArray(testArray[i]);
            probs[i] = Math.max(Math.min(ArrayUtil.max(cProbs), 1), 0);
        }
        return probs;
    }

    /**
     * Calculate the class (model index) that an instance is most likely to have
     * originated from.
     *
     * @param instance DataInstance object.
     * @return The index of the most likely model.
     */
    public int calcDataPointClass(DataInstance instance) {
        double[] cProbs = calcProbArray(instance);
        int cl = -1;
        double maxProb = -1;
        for (int c = 0; c < models.length; c++) {
            if (cProbs[c] > maxProb) {
                maxProb = cProbs[c];
                cl = c;
            }
        }
        return cl;
    }
}
