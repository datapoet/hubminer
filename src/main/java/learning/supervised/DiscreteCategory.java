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
package learning.supervised;

import data.representation.discrete.DiscretizedDataInstance;
import data.representation.discrete.DiscretizedDataSet;
import java.io.Serializable;
import java.util.ArrayList;

/**
 * This class represents a category of discretized data instances.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DiscreteCategory extends Category implements Serializable {
    
    private static final long serialVersionUID = 1L;

    private DiscretizedDataSet discDSet;
    public ArrayList<DiscretizedDataInstance> instances;
    private static final int DEFAULT_SIZE = 500;

    /**
     * The default constructor.
     */
    public DiscreteCategory() {
    }

    /**
     * Initialization.
     *
     * @param categoryName String that is the category name.
     * @param dset DataSet that is the data context.
     */
    public DiscreteCategory(String categoryName, DiscretizedDataSet dset) {
        setName(categoryName);
        this.discDSet = dset;
        indexes = new ArrayList<>(DEFAULT_SIZE);
    }

    /**
     * Initialization.
     *
     * @param categoryName String that is the category name.
     * @param dset DataSet object that is the data context.
     * @param initSize Integer that is the initialization size.
     */
    public DiscreteCategory(String categoryName, DiscretizedDataSet dset,
            int initSize) {
        setName(categoryName);
        this.discDSet = dset;
        indexes = new ArrayList<>(initSize);
    }

    /**
     * @return ArrayList<DiscretizedDataInstance> representing the data points
     * that belong to this data class.
     */
    public ArrayList<DiscretizedDataInstance> getData() {
        if (indexes == null && discDSet == null && discDSet.isEmpty()) {
            return null;
        }
        if (instances != null) {
            return instances;
        }
        ArrayList<DiscretizedDataInstance> data =
                new ArrayList<>(indexes.size());
        for (int i = 0; i < indexes.size(); i++) {
            data.add(discDSet.data.get(indexes.get(i)));
        }
        return data;
    }

    @Override
    public DiscreteCategory copy() {
        DiscreteCategory discCatCopy = new DiscreteCategory(getName(),
                discDSet);
        for (int i = 0; i < size(); i++) {
            discCatCopy.addInstanceIndex(indexes.get(i));
        }
        return discCatCopy;
    }

    /**
     * This method adds a data index and a corresponding data point to this
     * discretized category object.
     *
     * @param index Integer that is the index of the instance to add to the
     * category.
     */
    public void addInstanceIndex(int index) {
        if (indexes == null) {
            indexes = new ArrayList<>(DEFAULT_SIZE);
        }
        if (instances == null) {
            instances = new ArrayList<>(DEFAULT_SIZE);
        }
        indexes.add(index);
        if (discDSet != null) {
            instances.add(discDSet.getInstance(index));
        }
    }
}
