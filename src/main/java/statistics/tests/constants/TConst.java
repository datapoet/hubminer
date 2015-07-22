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
package statistics.tests.constants;

/**
 * This table-class contains the information on the critical values of Student's
 * t-statistic, based on the number of degrees of freedom. Up until 100 DoF are
 * currently supported.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class TConst {

    public float[] critVals5TwoTailed = null;
    public float[] critVals1TwoTailed = null;
    public float crit1TwoTailedInfty = 2.576f;
    public float crit5TwoTailedInfty = 1.960f;

    public TConst() {
        // Up to 100 degs of freedom for now.
        critVals5TwoTailed = new float[100];
        critVals1TwoTailed = new float[100];
        critVals1TwoTailed[0] = 63.66f; // For 1 degree of freedom.
        critVals1TwoTailed[1] = 9.925f;
        critVals1TwoTailed[2] = 5.841f;
        critVals1TwoTailed[3] = 4.604f;
        critVals1TwoTailed[4] = 4.032f;
        critVals1TwoTailed[5] = 3.707f;
        critVals1TwoTailed[6] = 3.499f;
        critVals1TwoTailed[7] = 3.355f;
        critVals1TwoTailed[8] = 3.250f;
        critVals1TwoTailed[9] = 3.169f;
        critVals1TwoTailed[10] = 3.106f;
        critVals1TwoTailed[11] = 3.055f;
        critVals1TwoTailed[12] = 3.012f;
        critVals1TwoTailed[13] = 2.977f;
        critVals1TwoTailed[14] = 2.947f;
        critVals1TwoTailed[15] = 2.921f;
        critVals1TwoTailed[16] = 2.898f;
        critVals1TwoTailed[17] = 2.878f;
        critVals1TwoTailed[18] = 2.861f;
        critVals1TwoTailed[19] = 2.845f;
        critVals1TwoTailed[20] = 2.831f;
        critVals1TwoTailed[21] = 2.819f;
        critVals1TwoTailed[22] = 2.807f;
        critVals1TwoTailed[23] = 2.797f;
        critVals1TwoTailed[24] = 2.787f;
        critVals1TwoTailed[25] = 2.779f;
        critVals1TwoTailed[26] = 2.771f;
        critVals1TwoTailed[27] = 2.763f;
        critVals1TwoTailed[28] = 2.756f;
        critVals1TwoTailed[29] = 2.750f;

        critVals1TwoTailed[30] = 2.745f;
        critVals1TwoTailed[31] = 2.741f;
        critVals1TwoTailed[32] = 2.736f;
        critVals1TwoTailed[33] = 2.731f;
        critVals1TwoTailed[34] = 2.727f;
        critVals1TwoTailed[35] = 2.722f;
        critVals1TwoTailed[36] = 2.718f;
        critVals1TwoTailed[37] = 2.713f;
        critVals1TwoTailed[38] = 2.708f;
        critVals1TwoTailed[39] = 2.704f;

        critVals1TwoTailed[40] = 2.702f;
        critVals1TwoTailed[41] = 2.700f;
        critVals1TwoTailed[42] = 2.697f;
        critVals1TwoTailed[43] = 2.695f;
        critVals1TwoTailed[44] = 2.693f;
        critVals1TwoTailed[45] = 2.691f;
        critVals1TwoTailed[46] = 2.689f;
        critVals1TwoTailed[47] = 2.686f;
        critVals1TwoTailed[48] = 2.684f;
        critVals1TwoTailed[49] = 2.682f;

        critVals1TwoTailed[50] = 2.680f;
        critVals1TwoTailed[51] = 2.678f;
        critVals1TwoTailed[52] = 2.675f;
        critVals1TwoTailed[53] = 2.673f;
        critVals1TwoTailed[54] = 2.671f;
        critVals1TwoTailed[55] = 2.669f;
        critVals1TwoTailed[56] = 2.667f;
        critVals1TwoTailed[57] = 2.664f;
        critVals1TwoTailed[58] = 2.662f;
        critVals1TwoTailed[59] = 2.660f;

        critVals1TwoTailed[60] = 2.659f;
        critVals1TwoTailed[61] = 2.659f;
        critVals1TwoTailed[62] = 2.658f;
        critVals1TwoTailed[63] = 2.657f;
        critVals1TwoTailed[64] = 2.656f;
        critVals1TwoTailed[65] = 2.656f;
        critVals1TwoTailed[66] = 2.655f;
        critVals1TwoTailed[67] = 2.654f;
        critVals1TwoTailed[68] = 2.653f;
        critVals1TwoTailed[69] = 2.653f;

        critVals1TwoTailed[70] = 2.652f;
        critVals1TwoTailed[71] = 2.652f;
        critVals1TwoTailed[72] = 2.651f;
        critVals1TwoTailed[73] = 2.650f;
        critVals1TwoTailed[74] = 2.649f;
        critVals1TwoTailed[75] = 2.649f;
        critVals1TwoTailed[76] = 2.648f;
        critVals1TwoTailed[77] = 2.647f;
        critVals1TwoTailed[78] = 2.646f;
        critVals1TwoTailed[79] = 2.646f;

        critVals1TwoTailed[80] = 2.645f;
        critVals1TwoTailed[81] = 2.644f;
        critVals1TwoTailed[82] = 2.644f;
        critVals1TwoTailed[83] = 2.643f;
        critVals1TwoTailed[84] = 2.642f;
        critVals1TwoTailed[85] = 2.641f;
        critVals1TwoTailed[86] = 2.641f;
        critVals1TwoTailed[87] = 2.640f;
        critVals1TwoTailed[88] = 2.639f;
        critVals1TwoTailed[89] = 2.639f;

        critVals1TwoTailed[90] = 2.638f;
        critVals1TwoTailed[91] = 2.637f;
        critVals1TwoTailed[92] = 2.636f;
        critVals1TwoTailed[93] = 2.636f;
        critVals1TwoTailed[94] = 2.635f;
        critVals1TwoTailed[95] = 2.634f;
        critVals1TwoTailed[96] = 2.633f;
        critVals1TwoTailed[97] = 2.633f;
        critVals1TwoTailed[98] = 2.632f;
        critVals1TwoTailed[99] = 2.631f;


        critVals5TwoTailed[0] = 12.706f;
        critVals5TwoTailed[1] = 4.303f;
        critVals5TwoTailed[2] = 3.182f;
        critVals5TwoTailed[3] = 2.776f;
        critVals5TwoTailed[4] = 2.571f;
        critVals5TwoTailed[5] = 2.447f;
        critVals5TwoTailed[6] = 2.365f;
        critVals5TwoTailed[7] = 2.306f;
        critVals5TwoTailed[8] = 2.262f;
        critVals5TwoTailed[9] = 2.228f;
        critVals5TwoTailed[10] = 2.201f;
        critVals5TwoTailed[11] = 2.179f;
        critVals5TwoTailed[12] = 2.16f;
        critVals5TwoTailed[13] = 2.145f;
        critVals5TwoTailed[14] = 2.131f;
        critVals5TwoTailed[15] = 2.12f;
        critVals5TwoTailed[16] = 2.11f;
        critVals5TwoTailed[17] = 2.101f;
        critVals5TwoTailed[18] = 2.093f;
        critVals5TwoTailed[19] = 2.086f;

        critVals5TwoTailed[20] = 2.08f;
        critVals5TwoTailed[21] = 2.074f;
        critVals5TwoTailed[22] = 2.069f;
        critVals5TwoTailed[23] = 2.064f;
        critVals5TwoTailed[24] = 2.06f;
        critVals5TwoTailed[25] = 2.056f;
        critVals5TwoTailed[26] = 2.052f;
        critVals5TwoTailed[27] = 2.048f;
        critVals5TwoTailed[28] = 2.045f;
        critVals5TwoTailed[29] = 2.042f;

        critVals5TwoTailed[30] = 2.04f;
        critVals5TwoTailed[31] = 2.038f;
        critVals5TwoTailed[32] = 2.036f;
        critVals5TwoTailed[33] = 2.034f;
        critVals5TwoTailed[34] = 2.032f;
        critVals5TwoTailed[35] = 2.029f;
        critVals5TwoTailed[36] = 2.027f;
        critVals5TwoTailed[37] = 2.025f;
        critVals5TwoTailed[38] = 2.023f;
        critVals5TwoTailed[39] = 2.021f;

        critVals5TwoTailed[40] = 2.02f;
        critVals5TwoTailed[41] = 2.019f;
        critVals5TwoTailed[42] = 2.018f;
        critVals5TwoTailed[43] = 2.017f;
        critVals5TwoTailed[44] = 2.016f;
        critVals5TwoTailed[45] = 2.015f;
        critVals5TwoTailed[46] = 2.014f;
        critVals5TwoTailed[47] = 2.013f;
        critVals5TwoTailed[48] = 2.012f;
        critVals5TwoTailed[49] = 2.010f;

        critVals5TwoTailed[50] = 2.009f;
        critVals5TwoTailed[51] = 2.008f;
        critVals5TwoTailed[52] = 2.007f;
        critVals5TwoTailed[53] = 2.006f;
        critVals5TwoTailed[54] = 2.005f;
        critVals5TwoTailed[55] = 2.004f;
        critVals5TwoTailed[56] = 2.003f;
        critVals5TwoTailed[57] = 2.002f;
        critVals5TwoTailed[58] = 2.001f;
        critVals5TwoTailed[59] = 2.000f;

        critVals5TwoTailed[60] = 2.000f;
        critVals5TwoTailed[61] = 1.999f;
        critVals5TwoTailed[62] = 1.999f;
        critVals5TwoTailed[63] = 1.999f;
        critVals5TwoTailed[64] = 1.998f;
        critVals5TwoTailed[65] = 1.998f;
        critVals5TwoTailed[66] = 1.998f;
        critVals5TwoTailed[67] = 1.997f;
        critVals5TwoTailed[68] = 1.997f;
        critVals5TwoTailed[69] = 1.997f;

        critVals5TwoTailed[70] = 1.996f;
        critVals5TwoTailed[71] = 1.996f;
        critVals5TwoTailed[72] = 1.996f;
        critVals5TwoTailed[73] = 1.995f;
        critVals5TwoTailed[74] = 1.995f;
        critVals5TwoTailed[75] = 1.995f;
        critVals5TwoTailed[76] = 1.994f;
        critVals5TwoTailed[77] = 1.994f;
        critVals5TwoTailed[78] = 1.993f;
        critVals5TwoTailed[79] = 1.993f;

        critVals5TwoTailed[80] = 1.993f;
        critVals5TwoTailed[81] = 1.993f;
        critVals5TwoTailed[82] = 1.992f;
        critVals5TwoTailed[83] = 1.992f;
        critVals5TwoTailed[84] = 1.992f;
        critVals5TwoTailed[85] = 1.991f;
        critVals5TwoTailed[86] = 1.991f;
        critVals5TwoTailed[87] = 1.991f;
        critVals5TwoTailed[88] = 1.990f;
        critVals5TwoTailed[89] = 1.990f;

        critVals5TwoTailed[90] = 1.990f;
        critVals5TwoTailed[91] = 1.989f;
        critVals5TwoTailed[92] = 1.989f;
        critVals5TwoTailed[93] = 1.989f;
        critVals5TwoTailed[94] = 1.988f;
        critVals5TwoTailed[95] = 1.988f;
        critVals5TwoTailed[96] = 1.988f;
        critVals5TwoTailed[97] = 1.987f;
        critVals5TwoTailed[98] = 1.987f;
        critVals5TwoTailed[99] = 1.987f;

    }
}
