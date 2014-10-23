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

import com.google.gson.Gson;
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
    
    @Override
    public String toString() {
        Gson gson = new Gson();
        StringBuilder sb = new StringBuilder();
        if (getIntegerMetric() != null) {
            sb.append("integers");
            sb.append(":");
            sb.append(getIntegerMetric().getClass().getCanonicalName());
            sb.append(":");
            String jsonString = gson.toJson(getIntegerMetric(),
                    getIntegerMetric().getClass());
            sb.append(jsonString);
            sb.append(" ");
        }
        if (getFloatMetric() != null) {
            sb.append("floats");
            sb.append(":");
            sb.append(getFloatMetric().getClass().getCanonicalName());
            sb.append(":");
            String jsonString = gson.toJson(getFloatMetric(),
                    getFloatMetric().getClass());
            sb.append(jsonString);
            sb.append(" ");
        }
        if (sparseMetric != null) {
            sb.append("sparse");
            sb.append(":");
            sb.append(sparseMetric.getClass().getCanonicalName());
            sb.append(":");
            String jsonString = gson.toJson(sparseMetric,
                    sparseMetric.getClass());
            sb.append(jsonString);
            sb.append(" ");
        }
        sb.append("\"combineBy\":");
        sb.append(getCombinationMethod());
        return sb.toString();
    }
    
    /**
     * This method obtains the object from a String representation.
     * @param stringRep String to obtain the object from, analogous to the one
     * produced by the toString() method.
     * @return SparseCombinedMetric object corresponding to the string.
     */
    public static SparseCombinedMetric fromString(String stringRep)
            throws Exception {
        SparseCombinedMetric scmet = new SparseCombinedMetric();
        String[] lineItems = stringRep.split("\\s+");
        if (lineItems.length == 2) {
            processMetricStringItem(scmet, lineItems[0]);
        } else if (lineItems.length == 3) {
            processMetricStringItem(scmet, lineItems[0]);
            processMetricStringItem(scmet, lineItems[1]);
        } else if (lineItems.length == 4) {
            processMetricStringItem(scmet, lineItems[0]);
            processMetricStringItem(scmet, lineItems[1]);
            processMetricStringItem(scmet, lineItems[2]);
        }
        switch (lineItems[lineItems.length - 1]) {
            case "\"combineBy\":SUM":
                scmet.setCombinationMethod(Mixer.SUM);
                break;
            case "\"combineBy\":PRODUCT":
                scmet.setCombinationMethod(Mixer.PRODUCT);
                break;
            case "\"combineBy\":MIN":
                scmet.setCombinationMethod(Mixer.MIN);
                break;
            case "\"combineBy\":MAX":
                scmet.setCombinationMethod(Mixer.MAX);
                break;
            case "\"combineBy\":AVERAGE":
                scmet.setCombinationMethod(Mixer.AVERAGE);
                break;
            case "\"combineBy\":EUCLIDEAN":
                scmet.setCombinationMethod(Mixer.EUCLIDEAN);
                break;
        }
        return scmet;
    }
    
    /**
     * Process the individual metric string when reading the
     * SparseCombinedMetric object from a string of the form that toString()
     * prints to. Internal method.
     * 
     * @param cmet SparseCombinedMetric object to load into.
     * @param metricStringItem String that is the individual metric String item.
     * @throws Exception 
     */
    private static void processMetricStringItem(SparseCombinedMetric cmet,
            String metricStringItem) throws Exception {
        Gson gson = new Gson();
        String[] metricLineItems = metricStringItem.split(":");
        String metricClassString = metricLineItems[1];
        Class metricClass = Class.forName(metricClassString);
        StringBuilder sb = new StringBuilder();
        for (int i = 2; i < metricLineItems.length; i++) {
            sb.append(metricLineItems[i]);
            if (i < metricLineItems.length - 1) {
                sb.append(":");
            }
        }
        String jsonString = sb.toString();
        Object metric = gson.fromJson(jsonString, metricClass);
        switch (metricLineItems[0]) {
            case "integers":
                cmet.setIntegerMetric((DistanceMeasure) metric);
                break;
            case "float":
                cmet.setFloatMetric((DistanceMeasure) metric);
                break;
            default:
                cmet.setSparseMetric((SparseMetric) metric);
                break;
        }
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
