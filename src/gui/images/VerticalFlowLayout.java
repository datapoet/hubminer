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
package gui.images;

import java.awt.LayoutManager;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.Container;
import java.awt.Insets;

/**
 * A vertical flow layout is similar to a flow layout but it layouts the
 * components vertically instead of horizontally. This class has been slightly
 * re-factored compared to the original source, in order to comply with the
 * style of the library. Re-factored by Nenad Tomasev.
 *
 * @author Vassili Dzuba.
 * @since March 24, 2001
 */
public class VerticalFlowLayout implements LayoutManager,
        java.io.Serializable {

    private int horizontalAlign;
    private int verticalAlign;
    private int horizontalGap;
    private int verticalGap;
    // Alignment constants.
    public final static int TOP = 0;
    public final static int CENTER = 1;
    public final static int BOTTOM = 2;
    public final static int LEFT = 3;
    public final static int RIGHT = 4;

    /**
     * Default constructor.
     */
    public VerticalFlowLayout() {
        this(CENTER, CENTER, 5, 5);
    }

    /**
     * @param horizontalAlign Integer that is the horizontal alignment type.
     * @param verticalAlign Integer that is the vertical alignment type.
     */
    public VerticalFlowLayout(int horizontalAlign, int verticalAlign) {
        this(horizontalAlign, verticalAlign, 5, 5);
    }

    /**
     * Initialization.
     *
     * @param horizontalAlign Integer that is the horizontal alignment type.
     * @param verticalAlign Integer that is the vertical alignment type.
     * @param horizontalGap Integer that is the horizontal gap.
     * @param verticalGap Integer that is the vertical gap.
     */
    public VerticalFlowLayout(int horizontalAlign, int verticalAlign,
            int horizontalGap, int verticalGap) {
        this.horizontalGap = horizontalGap;
        this.verticalGap = verticalGap;

        setAlignment(horizontalAlign, verticalAlign);
    }

    /**
     * Sets the alignment types.
     *
     * @param horizontalAlign Integer that is the horizontal alignment type.
     * @param verticalAlign Integer that is the vertical alignment type.
     */
    public final void setAlignment(int horizontalAlign, int verticalAlign) {
        this.horizontalAlign = horizontalAlign;
        this.verticalAlign = verticalAlign;
    }

    /**
     * @param horizontalGap Integer that is the horizontal gap.
     */
    public void setHgap(int horizontalGap) {
        this.horizontalGap = horizontalGap;
    }

    /**
     * @param verticalGap Integer that is the vertical gap.
     */
    public void setVgap(int verticalGap) {
        this.verticalGap = verticalGap;
    }

    /**
     * @return Integer that is the horizontal alignment type.
     */
    public int getHalignment() {
        return horizontalAlign;
    }

    /**
     * @return Integer that is the vertical alignment type.
     */
    public int getValignment() {
        return verticalAlign;
    }

    /**
     * @return Integer that is the horizontal gap.
     */
    public int getHgap() {
        return horizontalGap;
    }

    /**
     * @return Integer that is the vertical gap.
     */
    public int getVgap() {
        return verticalGap;
    }

    @Override
    public void addLayoutComponent(String name, Component comp) {
    }

    @Override
    public void removeLayoutComponent(Component comp) {
    }

    @Override
    public Dimension preferredLayoutSize(Container target) {
        synchronized (target.getTreeLock()) {
            Dimension dim = new Dimension(0, 0);
            int numComponents = target.getComponentCount();
            boolean firstVisibleComponent = true;
            for (int componentIndex = 0; componentIndex < numComponents;
                    componentIndex++) {
                Component comp = target.getComponent(componentIndex);
                if (comp.isVisible()) {
                    Dimension d = comp.getPreferredSize();
                    dim.width = Math.max(dim.width, d.width);
                    if (firstVisibleComponent) {
                        firstVisibleComponent = false;
                    } else {
                        dim.height += verticalGap;
                    }
                    dim.height += d.height;
                }
            }
            Insets insets = target.getInsets();
            dim.width += insets.left + insets.right + horizontalGap * 2;
            dim.height += insets.top + insets.bottom + verticalGap * 2;
            return dim;
        }
    }

    @Override
    public Dimension minimumLayoutSize(Container target) {
        synchronized (target.getTreeLock()) {
            Dimension dim = new Dimension(0, 0);
            int numComponents = target.getComponentCount();
            boolean firstVisibleComponent = true;
            for (int componentIndex = 0; componentIndex < numComponents;
                    componentIndex++) {
                Component comp = target.getComponent(componentIndex);
                if (comp.isVisible()) {
                    Dimension d = comp.getPreferredSize();
                    dim.width = Math.max(dim.width, d.width);
                    if (firstVisibleComponent) {
                        firstVisibleComponent = false;
                    } else {
                        dim.height += verticalGap;
                    }
                    dim.height += d.height;
                }
            }
            Insets insets = target.getInsets();
            dim.width += insets.left + insets.right + horizontalGap * 2;
            dim.height += insets.top + insets.bottom + verticalGap * 2;
            return dim;
        }
    }

    @Override
    public void layoutContainer(Container target) {
        synchronized (target.getTreeLock()) {
            Insets insets = target.getInsets();
            int maxHeight = target.getHeight()
                    - (insets.top + insets.bottom + verticalGap * 2);
            int numComponents = target.getComponentCount();
            int y = 0;
            Dimension preferredSize = preferredLayoutSize(target);
            Dimension targetSize = target.getSize();
            switch (verticalAlign) {
                case TOP:
                    y = insets.top;
                    break;
                case CENTER:
                    y = (targetSize.height - preferredSize.height) / 2;
                    break;
                case BOTTOM:
                    y = targetSize.height - preferredSize.height
                            - insets.bottom;
                    break;
            }
            for (int componentIndex = 0; componentIndex < numComponents;
                    componentIndex++) {
                Component comp = target.getComponent(componentIndex);
                if (comp.isVisible()) {
                    Dimension d = comp.getPreferredSize();
                    comp.setSize(d.width, d.height);
                    if ((y + d.height) <= maxHeight) {
                        if (y > 0) {
                            y += verticalGap;
                        }
                        int x = 0;
                        switch (horizontalAlign) {
                            case LEFT:
                                x = insets.left;
                                break;
                            case CENTER:
                                x = (targetSize.width - d.width) / 2;
                                break;
                            case RIGHT:
                                x = targetSize.width - d.width - insets.right;
                                break;
                        }
                        comp.setLocation(x, y);
                        y += d.getHeight();
                    } else {
                        break;
                    }
                }
            }
        }
    }

    @Override
    public String toString() {
        String horizontalAlignString = "";
        switch (horizontalAlign) {
            case TOP:
                horizontalAlignString = "top";
                break;
            case CENTER:
                horizontalAlignString = "center";
                break;
            case BOTTOM:
                horizontalAlignString = "bottom";
                break;
        }
        String verticalAlignString = "";
        switch (verticalAlign) {
            case TOP:
                verticalAlignString = "top";
                break;
            case CENTER:
                verticalAlignString = "center";
                break;
            case BOTTOM:
                verticalAlignString = "bottom";
                break;
        }
        return getClass().getName() + "[hgap=" + horizontalGap + ",vgap="
                + verticalGap + ",halign=" + horizontalAlignString + ",valign="
                + verticalAlignString + "]";
    }
}
