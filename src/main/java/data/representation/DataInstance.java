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
package data.representation;

import data.representation.util.DataMineConstants;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * The basic data class for representing data instances. The values are public
 * in the current implementation, in order to facilitate fast access and faster
 * development. The assumption is that the data is not modified by the learning
 * algorithms and that it is only written to during loading and preprocessing.
 * This is a reasonable assumption in learning evaluation systems. Therefore,
 * some care should be taken when working in a multi-threaded setting. In case
 * of applications where continuous data integrity can not be guaranteed, one
 * should wrap the data in a different container that would hide its fields.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DataInstance implements Serializable {

    private static final long serialVersionUID = 1L;
    // The DataSet that defines the feature types for this data instance.
    private DataSet dataContext = null;
    // The label of the data instance.
    private int category = 0;
    // Support for fuzzy labels is slowly being added throughout the code. Most
    // methods work with crisp labels, as is customary.
    public float[] fuzzyLabels = null;
    // Feature values.
    public int[] iAttr = null;
    public float[] fAttr = null;
    public String[] sAttr = null;
    // Identifier, which can be composed of multiple values and is thus also
    // represented as a data instance.
    private DataInstance identifier = null;

    /**
     * Noise is marked by having -1 label.
     *
     * @return True if not noise, false if noise.
     */
    public boolean notNoise() {
        return (category != -1);
    }

    /**
     * Noise is marked by having -1 label.
     *
     * @return True if noise, false if not noise.
     */
    public boolean isNoise() {
        return (category == -1);
    }

    /**
     * Sets the category of this instance to -1, which marks it as noise.
     */
    public void markAsNoise() {
        category = -1;
    }

    /**
     * In some I/O operations, an unchecked error might cause some instances to
     * be empty in a sense that they have all zero values. Not in this library,
     * but there were some cases with imported data. This method checks whether
     * an instance has all zero float values and is called in procedures that
     * check for data abnormalities in certain contexts. Of course, in a
     * different context, having all zero values might still yield a meaningful
     * representation.
     *
     * @return True if all float feature values are zero, false otherwise.
     */
    public boolean isZeroFloatVector() {
        if (hasFloatAtt()) {
            for (int i = 0; i < fAttr.length; i++) {
                if (fAttr[i] > 0) {
                    return false;
                }
            }
            return true;
        } else {
            return false;
        }
    }

    /**
     * @return True if the instance has float features, false otherwise.
     */
    public boolean hasFloatAtt() {
        if (fAttr == null || fAttr.length == 0) {
            return false;
        } else {
            return true;
        }
    }

    /**
     * @return True if the instance has integer features, false otherwise.
     */
    public boolean hasIntAtt() {
        if (iAttr == null || iAttr.length == 0) {
            return false;
        } else {
            return true;
        }
    }

    /**
     * @return True if the instance has nominal features, false otherwise.
     */
    public boolean hasNomAtt() {
        if (sAttr == null || sAttr.length == 0) {
            return false;
        } else {
            return true;
        }
    }

    /**
     * @return The number of float features of this instance.
     */
    public int getNumFAtt() {
        if (fAttr == null) {
            return 0;
        } else {
            return fAttr.length;
        }
    }

    /**
     * @return The number of integer features of this instance.
     */
    public int getNumIAtt() {
        if (iAttr == null) {
            return 0;
        } else {
            return iAttr.length;
        }
    }

    /**
     * @return The number of nominal features of this instance.
     */
    public int getNumNAtt() {
        if (sAttr == null) {
            return 0;
        } else {
            return sAttr.length;
        }
    }

    /**
     * Generates a comma-separated String that encodes all the float feature
     * values of this instance.
     *
     * @return
     */
    public String floatsToCSVString() {
        if (!hasFloatAtt()) {
            return "";
        }
        StringBuffer sb = new StringBuffer(160);
        sb.append(fAttr[0]);
        for (int i = 1; i < fAttr.length; i++) {
            sb.append(",");
            sb.append(fAttr[i]);
        }
        return sb.toString();
    }

    /**
     * Make a shallow data copy.
     *
     * @return DataInstance that is a shallow copy of the current instance.
     */
    public DataInstance makeArrayReferencedCopy() {
        DataInstance result = new DataInstance();
        result.embedInDataset(dataContext);
        result.setCategory(category);
        result.iAttr = iAttr;
        result.fAttr = fAttr;
        result.sAttr = sAttr;
        result.fuzzyLabels = fuzzyLabels;
        if (identifier != null) {
            result.setIdentifier(identifier.makeArrayReferencedCopy());
        }
        return result;
    }

    public DataInstance() {
    }

    /**
     * @param dataContext DataSet that contains the data definition for this
     * instance. This is used to instantiate feature value arrays.
     */
    public DataInstance(DataSet dataContext) {
        if (dataContext != null) {
            if (dataContext.hasIntAttr()) {
                iAttr = new int[dataContext.iAttrNames.length];
            }
            if (dataContext.hasFloatAttr()) {
                fAttr = new float[dataContext.fAttrNames.length];
            }
            if (dataContext.hasNominalAttr()) {
                sAttr = new String[dataContext.sAttrNames.length];
            }
            identifier = null;
            this.dataContext = dataContext;
        }
    }

    /**
     * @param identifier DataInstance holder for the identification info.
     * @param dataContext Data definition to implement in the instance.
     */
    public DataInstance(DataInstance identifier, DataSet dataContext) {
        this.identifier = identifier;
        this.dataContext = dataContext;
        if (dataContext != null) {
            if (dataContext.hasIntAttr()) {
                iAttr = new int[dataContext.iAttrNames.length];
            }
            if (dataContext.hasFloatAttr()) {
                fAttr = new float[dataContext.fAttrNames.length];
            }
            if (dataContext.hasNominalAttr()) {
                sAttr = new String[dataContext.sAttrNames.length];
            }
        }
    }

    /**
     * @return Category index.
     */
    public int getCategory() {
        return category;
    }

    /**
     * @param category Category index.
     */
    public void setCategory(int category) {
        this.category = category;
    }

    /**
     * @param identifier Data instance containing identifier data.
     */
    public void setIdentifier(DataInstance identifier) {
        this.identifier = identifier;
    }

    /**
     * @return The DataInstance containing the identification info on this
     * instance.
     */
    public DataInstance getIdentifier() {
        return this.identifier;
    }

    /**
     * @param dataContext DataSet to embed this instance in. A data definition
     * and a container.
     */
    public void embedInDataset(DataSet dataContext) {
        this.dataContext = dataContext;
    }

    /**
     * @return DataSet embedding this instance
     */
    public DataSet getEmbeddingDataset() {
        return this.dataContext;
    }

    /**
     * Checks if an instance belongs to a context that matches a certain data
     * definition.
     *
     * @param dset DataSet to match the definition of.
     * @return True if the contexts are essentially the same, false otherwise.
     */
    public boolean contextMatchesDataDefinition(DataSet dset) {
        return (dataContext == dset
                || dataContext.equalsInFeatureDefinition(dset));
    }

    /**
     * Make a copy of the current instance, but without identification data and
     * without class affiliation data.
     *
     * @return A copy of the current instance, without the identifier and
     * without the class label.
     * @throws Exception
     */
    public DataInstance copyContent() throws Exception {
        DataInstance instanceCopy = new DataInstance(dataContext);
        if (hasIntAtt()) {
            instanceCopy.iAttr = Arrays.copyOf(iAttr, iAttr.length);
        }
        if (hasFloatAtt()) {
            instanceCopy.fAttr = Arrays.copyOf(fAttr, fAttr.length);
        }
        if (hasNomAtt()) {
            instanceCopy.sAttr = Arrays.copyOf(sAttr, sAttr.length);
        }
        return instanceCopy;
    }
    
    /**
     * This method generates a copy of the identifier of the current instance.
     * 
     * @return DataInstance that is the copy of the identifier of the current
     * instance.
     */
    public DataInstance copyIdentifier() {
        if (identifier == null) {
            return null;
        } else {
            try {
                return identifier.copy();
            } catch (Exception e) {
                return null;
            }
        }
    }

    /**
     * A full copy method.
     *
     * @return DataInstance that is a copy of the current DataInstance.
     * @throws Exception
     */
    public DataInstance copy() throws Exception {
        DataInstance instanceCopy;
        try {
            instanceCopy = this.copyContent();
        } catch (Exception e) {
            instanceCopy = null;
            throw e;
        }
        instanceCopy.setCategory(getCategory());
        if (fuzzyLabels != null) {
            instanceCopy.fuzzyLabels = Arrays.copyOf(fuzzyLabels,
                    fuzzyLabels.length);
        }
        if (this.identifier != null) {
            DataInstance identifierCopy = this.identifier.copyContent();
            instanceCopy.setIdentifier(identifierCopy);
        }
        return instanceCopy;
    }

    /**
     * Tests for equality.
     *
     * @param instance DataInstance to compare with.
     * @return True if equal, false otherwise.
     */
    public boolean equals(DataInstance instance) {
        return (identifier.equalsByContent(instance.getIdentifier()))
                && (this.equalsByContent(instance)
                && category == instance.getCategory());
    }

    /**
     * Tests for equality of feature values.
     *
     * @param instance DataInstance to compare with.
     * @return True if equal, false otherwise.
     */
    public boolean equalsByContent(DataInstance instance) {
        for (int i = 0; i < getNumIAtt(); i++) {
            if (instance.iAttr[i] != iAttr[i]) {
                return false;
            }
        }
        for (int i = 0; i < getNumFAtt(); i++) {
            if (instance.fAttr[i] != fAttr[i]) {
                return false;
            }
        }
        for (int i = 0; i < getNumNAtt(); i++) {
            if (!instance.sAttr[i].equals(sAttr[i])) {
                return false;
            }
        }
        return true;
    }

    @Override
    public String toString() {
        StringBuffer s;
        try {
            s = new StringBuffer("");
            if (identifier != null) {
                s.append("<Identifier>");
                s.append(identifier.toString());
                s.append("</Identifier>");
            }
            s.append("<Instance>");
            s.append("<Label>");
            s.append(new Integer(category).toString());
            s.append("</Label>");
            for (int i = 0; i < getNumIAtt(); i++) {
                if (dataContext != null) {
                    s.append(dataContext.iAttrNames[i]);
                } else {
                    s.append("I_");
                    s.append(i);
                }
                s.append(":");
                if (DataMineConstants.isAcceptableInt(iAttr[i])) {
                    s.append(new Integer(iAttr[i]).toString());
                } else {
                    s.append("Unknown.");
                }
            }
            for (int i = 0; i < getNumFAtt(); i++) {
                if (dataContext != null) {
                    s.append(dataContext.fAttrNames[i]);
                } else {
                    s.append("F_");
                    s.append(i);
                }
                s.append(":");
                if (DataMineConstants.isAcceptableFloat(fAttr[i])) {
                    s.append(new Float(fAttr[i]).toString());
                } else {
                    s.append("Unknown.");
                }
            }
            for (int i = 0; i < getNumNAtt(); i++) {
                if (dataContext != null) {
                    s.append(dataContext.sAttrNames[i]);
                } else {
                    s.append("S_");
                    s.append(i);
                }
                s.append(":");
                s.append(sAttr[i]);
            }
            s.append("</Instance>");
            return s.toString();
        } catch (Exception e) {
            return e.toString();
        }
    }

    /**
     * Checks if there are some missing values in the instance value arrays.
     *
     * @return True if some values are marked as missing, false otherwise.
     * @throws Exception
     */
    public boolean hasMissingValues() throws Exception {
        for (int i = 0; i < getNumIAtt(); i++) {
            if (!DataMineConstants.isAcceptableInt(iAttr[i])) {
                return true;
            }
        }
        for (int i = 0; i < getNumFAtt(); i++) {
            if (!DataMineConstants.isAcceptableFloat(fAttr[i])) {
                return true;
            }
        }
        for (int i = 0; i < getNumNAtt(); i++) {
            if (sAttr[i] == null) {
                return true;
            }
        }
        return false;
    }

    /**
     * Multiply a DataInstance by a scalar value.
     *
     * @param alpha Scalar value, a float.
     * @param instance DataInstance.
     * @return A new, multiplied DataInstance.
     * @throws Exception
     */
    public static DataInstance multiplyByFactor(
            float alpha, DataInstance instance) throws Exception {
        if (instance == null) {
            throw new NullPointerException("Cannot multiply null instance.");
        }
        if (!DataMineConstants.isAcceptableFloat(alpha)) {
            throw new Exception("Invalid scalar: " + alpha);
        }
        DataInstance multipliedInstance = instance.copy();
        for (int i = 0; i < instance.getNumIAtt(); i++) {
            if (!DataMineConstants.isAcceptableInt(instance.iAttr[i])) {
                multipliedInstance.iAttr[i] = Integer.MAX_VALUE;
            } else {
                multipliedInstance.iAttr[i] = Math.round(
                        instance.iAttr[i] * alpha);
            }
        }
        for (int i = 0; i < instance.getNumFAtt(); i++) {
            if (!DataMineConstants.isAcceptableFloat(instance.fAttr[i])) {
                multipliedInstance.fAttr[i] = Float.MAX_VALUE;
            } else {
                multipliedInstance.fAttr[i] = instance.fAttr[i] * alpha;
            }
        }
        return multipliedInstance;
    }

    /**
     * Sums a list of DataInstance objects by summing up individual feature
     * values.
     *
     * @param instances ArrayList of DataInstance objects.
     * @return DataInstance representing the sum.
     */
    public static DataInstance sum(ArrayList<DataInstance> instances) {
        if (instances == null || instances.isEmpty()) {
            return null;
        }
        DataSet context = instances.get(0).dataContext;
        DataInstance sumInstance;
        if (context != null) {
            sumInstance = new DataInstance(context);
        } else {
            sumInstance = new DataInstance();
            sumInstance.fAttr = new float[instances.get(0).getNumFAtt()];
            sumInstance.iAttr = new int[instances.get(0).getNumIAtt()];
            sumInstance.sAttr = new String[instances.get(0).getNumNAtt()];
        }
        for (DataInstance instance : instances) {
            sumInstance.add(instance);
        }
        return sumInstance;
    }

    /**
     * Calculate the average of instances into an instance.
     *
     * @param instances An ArrayList of DataInstances.
     * @return DataInstance representing the average for all features.
     * @throws Exception
     */
    public static DataInstance average(ArrayList<DataInstance> instances)
            throws Exception {
        if (instances == null || instances.isEmpty()) {
            return null;
        }
        DataInstance result = sum(instances);
        DataInstance multRes = DataInstance.multiplyByFactor(
                1f / ((float) instances.size()), result);
        return multRes;
    }

    /**
     * Add a data instance values to the values of the current instance.
     *
     * @param instance DataInstance to add the values of.
     */
    public void add(DataInstance instance) {
        if (instance == null) {
            return;
        }
        for (int i = 0; i < getNumIAtt(); i++) {
            if (!DataMineConstants.isAcceptableInt(instance.iAttr[i])) {
                continue;
            } else {
                iAttr[i] += instance.iAttr[i];
            }
        }
        for (int i = 0; i < getNumFAtt(); i++) {
            if (!DataMineConstants.isAcceptableFloat(instance.fAttr[i])) {
                continue;
            } else {
                fAttr[i] += instance.fAttr[i];
            }
        }
    }
    
    /**
     * Add two DataInstance object values and return the result in another
     * DataInstance object.
     *
     * @param first DataInstance object.
     * @param second DataInstance object.
     * @return The sum of the two DataInstance objects, for all integer and
     * float feature values.
     */
    public static DataInstance add(DataInstance first, DataInstance second) {
        DataInstance result = new DataInstance(first.getEmbeddingDataset());
        result.add(first);
        result.add(second);
        return result;
    }

    /**
     * Subtract two DataInstance object values and return the result in another
     * DataInstance object.
     *
     * @param first DataInstance object.
     * @param second DataInstance object.
     * @return The difference of the two DataInstance objects, for all integer
     * and float feature values.
     */
    public static DataInstance subtract(DataInstance first,
            DataInstance second) throws Exception {
        DataInstance result = new DataInstance(first.getEmbeddingDataset());
        result.add(second);
        result = DataInstance.multiplyByFactor(-1, result);
        result.add(first);
        return result;
    }

    /**
     * Tests for equality by comparing float feature values.
     *
     * @param instance DataInstance object.
     * @return
     */
    public boolean equalsByFloatValue(DataInstance instance) {
        if (fAttr == null && instance.fAttr == null) {
            return true;
        }
        if (fAttr.length == 0 && instance.fAttr.length == 0) {
            return true;
        }
        if (fAttr.length != instance.fAttr.length) {
            return false;
        }
        int i = -1;
        while (++i < fAttr.length) {
            if ((fAttr[i] - instance.fAttr[i]) > DataMineConstants.EPSILON) {
                return false;
            }
        }
        return true;
    }

    /**
     * Tests for equality by comparing integer feature values.
     *
     * @param instance DataInstance object.
     * @return
     */
    public boolean equalsByIntegerValue(DataInstance instance) {
        if (iAttr == null && instance.iAttr == null) {
            return true;
        }
        if (iAttr.length == 0 && instance.iAttr.length == 0) {
            return true;
        }
        if (iAttr.length != instance.iAttr.length) {
            return false;
        }
        int i = -1;
        while (++i < iAttr.length) {
            if (iAttr[i] != instance.iAttr[i]) {
                return false;
            }
        }
        return true;
    }
}