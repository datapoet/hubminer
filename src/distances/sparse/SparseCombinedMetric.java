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
package distances.sparse;

import data.representation.DataInstance;
import data.representation.sparse.BOWInstance;
import data.representation.util.DataMineConstants;
import distances.primary.DistanceMeasure;
import distances.primary.CombinedMetric;

/**
 * This class implements the functionality for combining different parts of the
 * distances computed on sparse parts of the data representation.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SparseCombinedMetric extends CombinedMetric {

    public static final SparseCombinedMetric SPARSE_COSINE =
            new SparseCombinedMetric(null, null, new SparseCosineMetric(),
            Mixer.SUM);
    public static final SparseCombinedMetric SPARSE_MANHATTAN =
            new SparseCombinedMetric(null, null, new SparseManhattan(),
            Mixer.SUM);
    private SparseMetric sparseMetric;

    /**
     * The default constructor.
     */
    public SparseCombinedMetric() {
    }

    /**
     * @param integerMetric Metric for ints.
     * @param floatMetric Metric for floats.
     * @param sparseMetric Metric for sparse floats.
     * @param combMethod Mixer specifying how to combine individual metrics.
     */
    public SparseCombinedMetric(DistanceMeasure integerMetric,
            DistanceMeasure floatMetric, SparseMetric sparseMetric,
            Mixer combMethod) {
        super(integerMetric, floatMetric, combMethod);
        this.sparseMetric = sparseMetric;
    }

    /**
     * @return Currently used SparseMetric.
     */
    public SparseMetric getSparseMetric() {
        return sparseMetric;
    }

    /**
     * @param sparseMetric that will be used for sparse floats.
     */
    public void setSparseMetric(SparseMetric sparseMetric) {
        this.sparseMetric = sparseMetric;
    }

    /**
     * @param first Data instance.
     * @param second Data Instance.
     * @return The distance between data instances.
     */
    @Override
    public float dist(DataInstance first, DataInstance second)
            throws Exception {
        DistanceMeasure integerMetric = getIntegerMetric();
        DistanceMeasure floatMetric = getFloatMetric();
        Mixer combineBy = getCombinationMethod();
        float intDistance = integerMetric != null ? integerMetric.dist(
                first.iAttr, second.iAttr) : combineBy == Mixer.PRODUCT
                ? 1f : 0f;
        float floatDistance = floatMetric != null ? floatMetric.dist(
                first.fAttr, second.fAttr) : combineBy == Mixer.PRODUCT
                ? 1f : 0f;
        float sparseDistance = sparseMetric != null ? sparseMetric.dist(
                ((BOWInstance) first).getWordIndexesHash(),
                ((BOWInstance) second).getWordIndexesHash())
                : combineBy == Mixer.PRODUCT ? 1f : 0f;
        float totalDistance = combineBy == Mixer.PRODUCT ? 1f : 0f;
        switch (combineBy) {
            case SUM: {
                totalDistance += DataMineConstants.isAcceptableFloat(
                        intDistance) ? intDistance : 0;
                totalDistance += DataMineConstants.isAcceptableFloat(
                        floatDistance) ? floatDistance : 0;
                totalDistance += DataMineConstants.isAcceptableFloat(
                        sparseDistance) ? sparseDistance : 0;
                break;
            }
            case AVERAGE: {
                int counter = 0;
                if (DataMineConstants.isAcceptableFloat(intDistance)) {
                    counter++;
                    totalDistance += intDistance;
                }
                if (DataMineConstants.isAcceptableFloat(floatDistance)) {
                    counter++;
                    totalDistance += floatDistance;
                }
                if (DataMineConstants.isAcceptableFloat(sparseDistance)) {
                    counter++;
                    totalDistance += sparseDistance;
                }
                if (counter > 0) {
                    totalDistance /= counter;
                }
                break;
            }
            case MAX: {
                totalDistance = Math.max(totalDistance,
                        DataMineConstants.isAcceptableFloat(intDistance)
                        ? intDistance : 0);
                totalDistance = Math.max(totalDistance,
                        DataMineConstants.isAcceptableFloat(floatDistance)
                        ? floatDistance : 0);
                totalDistance = Math.max(totalDistance,
                        DataMineConstants.isAcceptableFloat(sparseDistance)
                        ? sparseDistance : 0);
                break;
            }
            case MIN: {
                totalDistance = Math.min(totalDistance,
                        DataMineConstants.isAcceptableFloat(intDistance)
                        ? intDistance : 0);
                totalDistance = Math.min(totalDistance,
                        DataMineConstants.isAcceptableFloat(floatDistance)
                        ? floatDistance : 0);
                totalDistance = Math.min(totalDistance,
                        DataMineConstants.isAcceptableFloat(sparseDistance)
                        ? sparseDistance : 0);
                break;
            }
            case PRODUCT: {
                totalDistance *= DataMineConstants.isAcceptableFloat(
                        intDistance) ? intDistance : 1;
                totalDistance *= DataMineConstants.isAcceptableFloat(
                        floatDistance) ? floatDistance : 1;
                totalDistance *= DataMineConstants.isAcceptableFloat(
                        sparseDistance) ? sparseDistance : 1;
                break;
            }
            case EUCLIDEAN: {
                totalDistance += DataMineConstants.isAcceptableFloat(
                        intDistance) ? Math.pow(intDistance, 2) : 0;
                totalDistance += DataMineConstants.isAcceptableFloat(
                        floatDistance) ? Math.pow(floatDistance, 2) : 0;
                totalDistance += DataMineConstants.isAcceptableFloat(
                        sparseDistance) ? Math.pow(sparseDistance, 2) : 0;
                totalDistance = (float) Math.sqrt(totalDistance);
                break;
            }
        }
        return totalDistance;
    }
}
