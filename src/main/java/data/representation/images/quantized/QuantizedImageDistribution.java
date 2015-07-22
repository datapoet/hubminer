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
package data.representation.images.quantized;

import data.representation.DataInstance;
import data.representation.DataSet;
import java.util.Arrays;

/**
 * Data holder class for quantized local feature distributions on an image. The
 * float features correspond to normalized frequencies of occurrences of
 * codebook matches. As this class is meant to represent a probability
 * distribution, all these normalized frequencies sum up to one. This class does
 * not implement any special methods or structures but is rather used for
 * dynamic typing in other modules, as an indicator that the underlying data is
 * in the appropriate format.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class QuantizedImageDistribution extends DataInstance {

    private String imagePath;

    /**
     * @param imagePath String that is the image path.
     */
    public void setPath(String imagePath) {
        this.imagePath = imagePath;
    }

    /**
     * @return String that is the image path.
     */
    public String getPath() {
        return imagePath;
    }

    /**
     * @param dset DataSet object to initialize the data context.
     */
    public QuantizedImageDistribution(DataSet dset) {
        super(dset);
    }

    @Override
    public QuantizedImageDistribution copyContent() throws Exception {
        QuantizedImageDistribution instanceCopy =
                new QuantizedImageDistribution(getEmbeddingDataset());
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
    public QuantizedImageDistribution copy() throws Exception {
        QuantizedImageDistribution instanceCopy;
        try {
            instanceCopy = this.copyContent();
        } catch (Exception e) {
            instanceCopy = null;
            throw e;
        }
        return instanceCopy;
    }
}
