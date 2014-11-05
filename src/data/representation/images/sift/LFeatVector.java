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
package data.representation.images.sift;

import data.representation.DataInstance;
import java.util.Arrays;

/**
 * A data holder class for the basic SIFT feature vector, where the first 4
 * values are Y, X , scale, orientation and the remaining ones belong to the
 * descriptor.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class LFeatVector extends DataInstance {

    public LFeatVector() {
    }

    /**
     * @param dataContext The embedding SIFTRepresentation container.
     */
    public LFeatVector(LFeatRepresentation dataContext) {
        super(dataContext);
        embedInDataset(dataContext);
    }

    /**
     *
     * @param allValues The float feature array of this particular SIFT vector.
     * @param dataContext The embedding SIFTRepresentation container.
     */
    public LFeatVector(float[] allValues, LFeatRepresentation dataContext) {
        fAttr = allValues;
        embedInDataset(dataContext);
    }

    /**
     * Initialize by a DataInstance.
     *
     * @param instance DataInstance to initialize this SIFT vector with.
     */
    public LFeatVector(DataInstance instance) {
        fAttr = instance.fAttr;
        iAttr = instance.iAttr;
        sAttr = instance.sAttr;
        embedInDataset(instance.getEmbeddingDataset());
        setIdentifier(instance.getIdentifier());
    }

    /**
     * @param dataContext The embedding SIFTRepresentation container.
     */
    public void setContext(LFeatRepresentation dataContext) {
        embedInDataset(dataContext);
    }

    /**
     * @return The X coordinate of the feature in the image.
     */
    public float getX() {
        if (fAttr != null && getNumFAtt() > 1) {
            return fAttr[1];
        } else {
            return -1.f;
        }
    }

    /**
     * @param val The X coordinate of the feature in the image.
     */
    public void setX(float val) {
        if (fAttr != null && getNumFAtt() > 1) {
            fAttr[1] = val;
        }
    }

    /**
     * @return The Y coordinate of the feature in the image.
     */
    public float getY() {
        if (fAttr != null && getNumFAtt() > 0) {
            return fAttr[0];
        } else {
            return -1.f;
        }
    }

    /**
     * @param val The Y coordinate of the feature in the image.
     */
    public void setY(float val) {
        if (fAttr != null && getNumFAtt() > 0) {
            fAttr[0] = val;
        }
    }

    /**
     * @return The scale of the feature in the image.
     */
    public float getScale() {
        if (fAttr != null && getNumFAtt() > 2) {
            return fAttr[2];
        } else {
            return -1.f;
        }
    }

    /**
     * @param val The scale of the feature in the image.
     */
    public void setScale(float val) {
        if (fAttr != null && getNumFAtt() > 2) {
            fAttr[2] = val;
        }
    }

    /**
     * @return The orientation of the feature in the image.
     */
    public float getAngle() {
        if (fAttr != null && getNumFAtt() > 3) {
            return fAttr[3];
        } else {
            return -1.f;
        }
    }

    /**
     * @param val The orientation of the feature in the image.
     */
    public void setAngle(float val) {
        if (fAttr != null && getNumFAtt() > 3) {
            fAttr[3] = val;
        }
    }

    /**
     * Make a CSV string out of the current representation.
     *
     * @return
     */
    public String toCSVString() {
        StringBuffer sb = new StringBuffer(160);
        sb.append(fAttr[0]);
        for (int i = 1; i < fAttr.length; i++) {
            sb.append(",");
            sb.append(fAttr[i]);
        }
        return sb.toString();
    }

    /**
     * Read a CSV string representing a SIFTVector representation.
     *
     * @param inputString String in CSV format representing a SIFTVector.
     */
    public void fillFromCSVString(String inputString) {
        String[] parts = inputString.split(",");
        fAttr = new float[parts.length];
        for (int i = 0; i < parts.length; i++) {
            fAttr[i] = Float.parseFloat(parts[i]);
        }
    }

    @Override
    public LFeatVector copyContent() throws Exception {
        LFeatVector instanceCopy = new LFeatVector();
        instanceCopy.embedInDataset(getEmbeddingDataset());
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

    @Override
    public LFeatVector copy() throws Exception {
        LFeatVector instanceCopy;
        try {
            instanceCopy = this.copyContent();
        } catch (Exception e) {
            instanceCopy = null;
            throw e;
        }
        return instanceCopy;
    }
}