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
package filters;

import data.representation.DataSet;

/**
 * This interface declares basic filtering operations on the data. It either
 * performs the filtering by modifying the existing dataset or by making a new
 * copy.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public interface FilterInterface {

    /**
     * Filter the data and make a new copy.
     *
     * @param dset DataSet to filter.
     * @return Filtered dataset.
     */
    public DataSet filterAndCopy(DataSet dset);

    /**
     * Performs filtering on the data.
     *
     * @param dset Filtered dataset.
     */
    public void filter(DataSet dset);
}
