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
package data.neighbors;

/**
 * This interface defines passing of the kNN sets and the associated statistics
 * within the experimental framework.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public interface NSFUserInterface {

    /**
     * @param nsf NeighborSetFinder object for calculating and holding kNN sets.
     */
    public void setNSF(NeighborSetFinder nsf);

    /**
     * @return NeighborSetFinder object for calculating and holding kNN sets.
     */
    public NeighborSetFinder getNSF();

    /**
     * Indicating that the object is shared and no recalculations can be done on
     * it for synchronization purposes. An empty implementation is appropriate
     * in methods that do not modify the NeighborSetFinder object's content.
     */
    public void noRecalcs();
    
    /**
     * This method asks the NSF user to declare its preferred neighborhood size.
     * 
     * @return Integer that is the neighborhood size needed for NSF
     * calculations. 
     */
    public int getNeighborhoodSize();
}
