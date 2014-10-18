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

import data.representation.DataSet;
import java.io.Serializable;
import java.util.ArrayList;
import learning.unsupervised.Cluster;

/**
 * This class represents a data category, a labeled data cluster.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class Category extends Cluster implements Serializable {
    
    private static final long serialVersionUID = 1L;

    private String name;

    /**
     * The default constructor.
     */
    public Category() {
    }

    /**
     * Initialization.
     *
     * @param categoryName String that is the category name.
     * @param dset DataSet object that is the data context.
     */
    public Category(String categoryName, DataSet dset) {
        super(dset);
        this.name = categoryName;
    }

    /**
     * Initialization.
     *
     * @param categoryName String that is the category name.
     * @param dset DataSet object that is the data context.
     * @param dataPoints ArrayList of data points.
     */
    public Category(String categoryName, DataSet dset, ArrayList dataPoints) {
        super(dset, dataPoints);
        this.name = categoryName;
    }

    /**
     * Initialization.
     *
     * @param categoryName String that is the category name.
     * @param dset DataSet object that is the data context.
     * @param dataPoints ArrayList of data points.
     * @param dataIndexes ArrayList<Integer> of data indexes belonging to the
     * category cluster.
     */
    public Category(String categoryName, DataSet dset, ArrayList dataPoints,
            ArrayList<Integer> dataIndexes) {
        super(dset, dataIndexes);
        this.name = categoryName;
    }

    /**
     * Initialization.
     *
     * @param categoryName String that is the category name.
     * @param initSize Integer that is the initial cluster size.
     * @param dset DataSet object that is the data context.
     */
    public Category(String categoryName, int initSize, DataSet dset) {
        super(dset, initSize);
        this.name = categoryName;
    }

    @Override
    public Category copy() {
        return new Category(name, getDefinitionDataset(),
                (ArrayList<Integer>) indexes.clone());
    }

    /**
     * @return String that is the category name.
     */
    public String getName() {
        return name;
    }

    /**
     * @param categoryName String that is the category name.
     */
    public void setName(String categoryName) {
        this.name = categoryName;
    }

    /**
     * This method counts the total number of elements in a category array.
     *
     * @param categoryArray Category[] representing a category array.
     * @return Integer that is the total number of points in all the categories
     * from the array parameter.
     */
    public static int getNumPointsInAllCategories(Category[] categoryArray) {
        if (categoryArray == null || categoryArray.length == 0) {
            return 0;
        }
        int totalElements = 0;
        for (Category cat : categoryArray) {
            if (cat != null) {
                totalElements += cat.size();
            }
        }
        return totalElements;
    }
}