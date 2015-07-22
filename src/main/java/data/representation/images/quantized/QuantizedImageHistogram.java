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
 * Data holder for representing quantized image histograms as integer counts.
 * This class does not implement any special or additional structures and
 * methods, but is rather used for dynamic typing.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class QuantizedImageHistogram extends DataInstance {

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
    public QuantizedImageHistogram(DataSet dset) {
        super(dset);
    }

    /**
     * @param instance DataInstance object to initialize the histograms with.
     */
    public QuantizedImageHistogram(DataInstance instance) {
        super(instance.getEmbeddingDataset());
        this.fAttr = instance.fAttr;
        this.iAttr = instance.iAttr;
        this.sAttr = instance.sAttr;
        this.setCategory(instance.getCategory());
        this.fuzzyLabels = instance.fuzzyLabels;
    }

    @Override
    public QuantizedImageHistogram copyContent() throws Exception {
        QuantizedImageHistogram instanceCopy =
                new QuantizedImageHistogram(getEmbeddingDataset());
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
    public QuantizedImageHistogram copy() throws Exception {
        QuantizedImageHistogram instanceCopy;
        try {
            instanceCopy = this.copyContent();
        } catch (Exception e) {
            instanceCopy = null;
            throw e;
        }
        return instanceCopy;
    }
}
