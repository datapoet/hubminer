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
package distances.primary;

import com.google.gson.Gson;
import data.representation.DataInstance;
import data.representation.util.DataMineConstants;
import java.io.Serializable;

/**
 * This class implements a way to combine the distances between the integer and
 * float parts of the data representation.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class CombinedMetric implements Serializable {

    private static final long serialVersionUID = 1L;
    // These are the possible combination methods.
    public enum Mixer {

        SUM, AVERAGE, MAX, MIN, PRODUCT, EUCLIDEAN
    }
    public static final Mixer DEFAULT = Mixer.SUM;
    public static final CombinedMetric EUCLIDEAN = new CombinedMetric(
            new MinkowskiMetric(), new MinkowskiMetric(), DEFAULT);
    public static final CombinedMetric MANHATTAN = new CombinedMetric(
            new Manhattan(), new Manhattan(), DEFAULT);
    public static final CombinedMetric FLOAT_EUCLIDEAN = new CombinedMetric(
            null, new MinkowskiMetric(), DEFAULT);
    public static final CombinedMetric FLOAT_MANHATTAN = new CombinedMetric(
            null, new Manhattan(), DEFAULT);
    public static final CombinedMetric FLOAT_COSINE = new CombinedMetric(
            null, new CosineMetric(), DEFAULT);
    public static final CombinedMetric FLOAT_TANIMOTO = new CombinedMetric(
            null, new TanimotoDistance(), DEFAULT);
    public static final CombinedMetric FLOAT_KL = new CombinedMetric(
            null, new SymmetricKLDivergence(), DEFAULT);
    public static final CombinedMetric FLOAT_BRAY_CURTIS = new CombinedMetric(
            null, new BrayCurtis(), DEFAULT);
    public static final CombinedMetric FLOAT_CANBERRA = new CombinedMetric(
            null, new Canberra(), DEFAULT);
    public static final CombinedMetric INT_EUCLIDEAN = new CombinedMetric(
            new MinkowskiMetric(), null, DEFAULT);
    public static final CombinedMetric INT_MANHATTAN = new CombinedMetric(
            new Manhattan(), null, DEFAULT);
    public static final CombinedMetric INT_COSINE = new CombinedMetric(
            new CosineMetric(), null, DEFAULT);
    public static final CombinedMetric INT_TANIMOTO = new CombinedMetric(
            new TanimotoDistance(), null, DEFAULT);
    public static final CombinedMetric INT_BRAY_CURTIS = new CombinedMetric(
            new BrayCurtis(), null, DEFAULT);
    public static final CombinedMetric INT_CANBERRA = new CombinedMetric(
            new Canberra(), null, DEFAULT);
    private DistanceMeasure integerMetric;
    private DistanceMeasure floatMetric;
    private Mixer combineBy;

    public CombinedMetric() {
    }

    /**
     * @param integerMetric DistanceMeasure calculating int distances.
     * @param floatMetric DistanceMeasure calculating float distances.
     * @param combineBy Mixer method for combining int and float distances.
     */
    public CombinedMetric(DistanceMeasure integerMetric,
            DistanceMeasure floatMetric, Mixer combineBy) {
        this.integerMetric = integerMetric;
        this.floatMetric = floatMetric;
        this.combineBy = combineBy;
    }
    
    @Override
    public String toString() {
        Gson gson = new Gson();
        StringBuilder sb = new StringBuilder();
        if (integerMetric != null) {
            sb.append("integers");
            sb.append(":");
            sb.append(integerMetric.getClass().getCanonicalName());
            sb.append(":");
            String jsonString = gson.toJson(integerMetric,
                    integerMetric.getClass());
            sb.append(jsonString);
            sb.append(" ");
        }
        if (floatMetric != null) {
            sb.append("floats");
            sb.append(":");
            sb.append(floatMetric.getClass().getCanonicalName());
            sb.append(":");
            String jsonString = gson.toJson(floatMetric,
                    floatMetric.getClass());
            sb.append(jsonString);
            sb.append(" ");
        }
        sb.append("\"combineBy\":");
        sb.append(combineBy);
        return sb.toString();
    }
    
    /**
     * This method obtains the object from a String representation.
     * @param stringRep String to obtain the object from, analogous to the one
     * produced by the toString() method.
     * @return CombinedMetric object corresponding to the string.
     */
    public static CombinedMetric fromString(String stringRep) throws Exception {
        CombinedMetric cmet = new CombinedMetric();
        String[] lineItems = stringRep.split("\\s+");
        if (lineItems.length == 2) {
            processMetricStringItem(cmet, lineItems[0]);
        } else if (lineItems.length == 3) {
            processMetricStringItem(cmet, lineItems[0]);
            processMetricStringItem(cmet, lineItems[1]);
        }
        switch (lineItems[lineItems.length - 1]) {
            case "\"combineBy\":SUM":
                cmet.combineBy = Mixer.SUM;
                break;
            case "\"combineBy\":PRODUCT":
                cmet.combineBy = Mixer.PRODUCT;
                break;
            case "\"combineBy\":MIN":
                cmet.combineBy = Mixer.MIN;
                break;
            case "\"combineBy\":MAX":
                cmet.combineBy = Mixer.MAX;
                break;
            case "\"combineBy\":AVERAGE":
                cmet.combineBy = Mixer.AVERAGE;
                break;
            case "\"combineBy\":EUCLIDEAN":
                cmet.combineBy = Mixer.EUCLIDEAN;
                break;
        }
        return cmet;
    }
    
    /**
     * Process the individual metric string when reading the CombinedMetric
     * object from a string of the form that toString() prints to. Internal
     * method.
     * 
     * @param cmet CombinedMetric object to load into.
     * @param metricStringItem String that is the individual metric String item.
     * @throws Exception 
     */
    private static void processMetricStringItem(CombinedMetric cmet,
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
        if (metricLineItems[0].equals("integers")) {
            cmet.integerMetric = (DistanceMeasure) metric;
        } else {
            cmet.floatMetric = (DistanceMeasure) metric;
        }
    }

    /**
     * @return The selected Mixer combination method.
     */
    public Mixer getCombinationMethod() {
        return combineBy;
    }

    /**
     * @param combineBy Mixer combination method to combine the metrics by.
     */
    public void setCombinationMethod(Mixer combineBy) {
        this.combineBy = combineBy;
    }

    /**
     * @return DistanceMeasure that is currently selected for ints.
     */
    public DistanceMeasure getIntegerMetric() {
        return integerMetric;
    }

    /**
     * @param DistanceMeasure to use for ints.
     */
    public void setIntegerMetric(DistanceMeasure integerMetric) {
        this.integerMetric = integerMetric;
    }

    /**
     * @return DistanceMeasure that is currently selected for floats.
     */
    public DistanceMeasure getFloatMetric() {
        return floatMetric;
    }

    /**
     * @param DistanceMeasure to use for floats.
     */
    public void setFloatMetric(DistanceMeasure floatMetric) {
        this.floatMetric = floatMetric;
    }

    /**
     * @param first Data instance.
     * @param second Data Instance.
     * @return The distance between data instances.
     */
    public float dist(DataInstance first, DataInstance second)
            throws Exception {
        boolean hasIntAtt = first.hasIntAtt() && second.hasIntAtt();
        boolean hasFloatAtt = first.hasFloatAtt() && second.hasFloatAtt();
        float intDistance = integerMetric != null && hasIntAtt ?
                integerMetric.dist(first.iAttr, second.iAttr) :
                combineBy == Mixer.PRODUCT ? 1f : 0f;
        float floatDistance = floatMetric != null && hasFloatAtt ?
                floatMetric.dist(first.fAttr, second.fAttr) :
                combineBy == Mixer.PRODUCT ? 1f : 0f;
        float totalDistance = combineBy == Mixer.PRODUCT ? 1f : 0f;
        switch (combineBy) {
            case SUM: {
                totalDistance += DataMineConstants.isAcceptableFloat(
                        intDistance) ? intDistance : 0;
                totalDistance += DataMineConstants.isAcceptableFloat(
                        floatDistance) ? floatDistance : 0;
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
                break;
            }
            case MIN: {
                totalDistance = Math.min(totalDistance,
                        DataMineConstants.isAcceptableFloat(intDistance)
                        ? intDistance : 0);
                totalDistance = Math.min(totalDistance,
                        DataMineConstants.isAcceptableFloat(floatDistance)
                        ? floatDistance : 0);
                break;
            }
            case PRODUCT: {
                totalDistance *= DataMineConstants.isAcceptableFloat(
                        intDistance) ? intDistance : 1;
                totalDistance *= DataMineConstants.isAcceptableFloat(
                        floatDistance) ? floatDistance : 1;
                break;
            }
            case EUCLIDEAN: {
                totalDistance += DataMineConstants.isAcceptableFloat(
                        intDistance) ? Math.pow(intDistance, 2) : 0;
                totalDistance += DataMineConstants.isAcceptableFloat(
                        floatDistance) ? Math.pow(floatDistance, 2) : 0;
                totalDistance = (float) Math.sqrt(totalDistance);
                break;
            }
        }
        return totalDistance;
    }
}