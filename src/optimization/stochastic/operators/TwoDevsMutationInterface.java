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
package optimization.stochastic.operators;

/**
 * This interface sets up the methods for handling both large and small
 * mutations simultaneously.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public interface TwoDevsMutationInterface extends MutationInterface {

    /**
     * Sets the probability of small mutations occurring.
     *
     * @param pSmall Probability of small mutations.
     */
    public void setPSmall(float pSmall);

    /**
     * @return Float value that is the probability of small mutations occurring.
     */
    public float getPSmall();

    /**
     * @param stDevSmall Standard deviation of small mutations in a homogenous
     * mutation scheme.
     */
    public void setDevSmall(float stDevSmall);

    /**
     * @return Float value representing the standard deviation of small
     * mutations in a homogenous mutation scheme.
     */
    public float getDevSmall();

    /**
     * @param stDevBig Standard deviation of large mutations in a homogenous
     * mutation scheme.
     */
    public void setDevBig(float stDevBig);

    /**
     * @return Float value representing the standard deviation of large
     * mutations in a homogenous mutation scheme.
     */
    public float getDevBig();

    /**
     * @param stDevSmall An array of standard deviations of small mutations in a
     * heterogenous mutation scheme.
     */
    public void setDevsSmall(float[] stDevSmall);

    /**
     * @return An array of float values representing the standard deviations of
     * small mutations in a heterogenous mutation scheme.
     */
    public float[] getDevsSmall();

    /**
     * @param stDevBig An array of standard deviations of large mutations in a
     * heterogenous mutation scheme.
     */
    public void setDevsBig(float[] stDevBig);

    /**
     * @return An array of float values representing the standard deviations of
     * large mutations in a heterogenous mutation scheme.
     */
    public float[] getDevsBig();
}
