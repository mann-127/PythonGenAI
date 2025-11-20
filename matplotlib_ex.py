import matplotlib.pyplot as plt
import numpy as np
 
# ============================================
# 1. LINE PLOT
# ============================================
def line_plot():
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    plt.figure(figsize=(10, 5))
    plt.plot(x, y1, 'b-', label='sin(x)', linewidth=2)
    plt.plot(x, y2, 'r--', label='cos(x)', linewidth=2)
    plt.title('Line Plot - Sine and Cosine Waves')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
 
 
# ============================================
# 2. BAR CHART
# ============================================
def bar_chart():
    categories = ['A', 'B', 'C', 'D', 'E']
    values = [23, 45, 56, 78, 32]
    
    plt.figure(figsize=(10, 5))
    plt.bar(categories, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
    plt.title('Vertical Bar Chart')
    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.grid(axis='y', alpha=0.3)
    plt.show()
 
 
# ============================================
# 3. HORIZONTAL BAR CHART
# ============================================
def horizontal_bar():
    categories = ['Product A', 'Product B', 'Product C', 'Product D']
    values = [150, 200, 175, 225]
    
    plt.figure(figsize=(10, 5))
    plt.barh(categories, values, color='steelblue')
    plt.title('Horizontal Bar Chart - Sales by Product')
    plt.xlabel('Sales ($)')
    plt.ylabel('Products')
    plt.grid(axis='x', alpha=0.3)
    plt.show()
 
 
# ============================================
# 4. SCATTER PLOT
# ============================================
def scatter_plot():
    x = np.random.rand(50) * 100
    y = np.random.rand(50) * 100
    colors = np.random.rand(50)
    sizes = np.random.rand(50) * 500
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap='viridis')
    plt.colorbar(label='Color Scale')
    plt.title('Scatter Plot with Variable Size and Color')
    plt.xlabel('X Values')
    plt.ylabel('Y Values')
    plt.grid(True, alpha=0.3)
    plt.show()
 
 
# ============================================
# 5. PIE CHART
# ============================================
def pie_chart():
    labels = ['Python', 'Java', 'JavaScript', 'C++', 'Ruby']
    sizes = [35, 25, 20, 15, 5]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
    explode = (0.1, 0, 0, 0, 0)  # Explode 1st slice
    
    plt.figure(figsize=(10, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    plt.title('Programming Languages Distribution')
    plt.axis('equal')
    plt.show()
 
 
# ============================================
# 6. HISTOGRAM
# ============================================
def histogram():
    data = np.random.randn(1000)
    
    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Histogram - Normal Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.3)
    plt.show()
 
 
# ============================================
# 7. BOX PLOT
# ============================================
def box_plot():
    data1 = np.random.normal(100, 10, 200)
    data2 = np.random.normal(90, 20, 200)
    data3 = np.random.normal(80, 30, 200)
    data4 = np.random.normal(70, 40, 200)
    
    data = [data1, data2, data3, data4]
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, tick_labels=['Group A', 'Group B', 'Group C', 'Group D'])
    plt.title('Box Plot - Distribution Comparison')
    plt.ylabel('Values')
    plt.grid(axis='y', alpha=0.3)
    plt.show()
 
 
# ============================================
# 8. AREA PLOT
# ============================================
def area_plot():
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    plt.figure(figsize=(10, 5))
    plt.fill_between(x, y1, alpha=0.5, label='sin(x)')
    plt.fill_between(x, y2, alpha=0.5, label='cos(x)')
    plt.title('Area Plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# ============================================
# 9. STACKED BAR CHART
# ============================================
def stacked_bar():
    categories = ['Q1', 'Q2', 'Q3', 'Q4']
    product_a = [20, 35, 30, 35]
    product_b = [25, 32, 34, 20]
    product_c = [15, 20, 25, 30]
    
    x = np.arange(len(categories))
    width = 0.6
    
    plt.figure(figsize=(10, 6))
    plt.bar(x, product_a, width, label='Product A', color='#FF6B6B')
    plt.bar(x, product_b, width, bottom=product_a, label='Product B', color='#4ECDC4')
    plt.bar(x, product_c, width, bottom=np.array(product_a)+np.array(product_b),
            label='Product C', color='#45B7D1')
    
    plt.title('Stacked Bar Chart - Quarterly Sales')
    plt.xlabel('Quarter')
    plt.ylabel('Sales')
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()
 
 
# ============================================
# 10. GROUPED BAR CHART
# ============================================
def grouped_bar():
    categories = ['2020', '2021', '2022', '2023']
    product_a = [20, 35, 30, 35]
    product_b = [25, 32, 34, 20]
    product_c = [15, 20, 25, 30]
    
    x = np.arange(len(categories))
    width = 0.25
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width, product_a, width, label='Product A', color='#FF6B6B')
    plt.bar(x, product_b, width, label='Product B', color='#4ECDC4')
    plt.bar(x + width, product_c, width, label='Product C', color='#45B7D1')
    
    plt.title('Grouped Bar Chart - Yearly Comparison')
    plt.xlabel('Year')
    plt.ylabel('Sales (in thousands)')
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()
 
 
# ============================================
# 11. HEATMAP
# ============================================
def heatmap():
    data = np.random.rand(10, 10)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap='YlOrRd', aspect='auto')
    plt.colorbar(label='Intensity')
    plt.title('Heatmap')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.show()
 
 
# ============================================
# 12. CONTOUR PLOT
# ============================================
def contour_plot():
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    
    plt.figure(figsize=(10, 8))
    contour = plt.contour(X, Y, Z, levels=15, cmap='viridis')
    plt.colorbar(contour, label='Z Values')
    plt.title('Contour Plot')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
 
 
# ============================================
# 13. 3D SURFACE PLOT
# ============================================
def surface_3d():
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    
    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.8)
    ax.set_title('3D Surface Plot')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.colorbar(surf, shrink=0.5)
    plt.show()


# ============================================
# 14. VIOLIN PLOT
# ============================================
def violin_plot():
    data1 = np.random.normal(100, 10, 200)
    data2 = np.random.normal(90, 20, 200)
    data3 = np.random.normal(80, 15, 200)
    
    data = [data1, data2, data3]
    
    plt.figure(figsize=(10, 6))
    parts = plt.violinplot(data, positions=[1, 2, 3], showmeans=True)
    plt.title('Violin Plot - Distribution Shape')
    plt.xlabel('Groups')
    plt.ylabel('Values')
    plt.xticks([1, 2, 3], ['Group A', 'Group B', 'Group C'])
    plt.grid(axis='y', alpha=0.3)
    plt.show()
 
 
# ============================================
# 15. STEP PLOT
# ============================================
def step_plot():
    x = np.linspace(0, 10, 20)
    y = np.sin(x)
    
    plt.figure(figsize=(10, 5))
    plt.step(x, y, where='mid', linewidth=2, label='Step Plot')
    plt.plot(x, y, 'o', color='red', markersize=5, label='Data Points')
    plt.title('Step Plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
 
 
# ============================================
# 16. ERROR BAR PLOT
# ============================================
def error_bar():
    x = np.arange(0, 10, 1)
    y = x ** 2
    error = np.random.rand(10) * 10
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(x, y, yerr=error, fmt='o-', capsize=5, capthick=2,
                 ecolor='red', linewidth=2, markersize=8)
    plt.title('Error Bar Plot')
    plt.xlabel('X Values')
    plt.ylabel('Y Values')
    plt.grid(True, alpha=0.3)
    plt.show()
 
 
# ============================================
# 17. POLAR PLOT
# ============================================
def polar_plot():
    theta = np.linspace(0, 2 * np.pi, 100)
    r = np.abs(np.sin(3 * theta))
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection='polar')
    ax.plot(theta, r, linewidth=2)
    ax.fill(theta, r, alpha=0.3)
    ax.set_title('Polar Plot')
    plt.show()
 
 
# ============================================
# 18. STEM PLOT
# ============================================
def stem_plot():
    x = np.linspace(0, 10, 20)
    y = np.sin(x)
    
    plt.figure(figsize=(10, 5))
    plt.stem(x, y, linefmt='blue', markerfmt='ro', basefmt='gray')
    plt.title('Stem Plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    plt.show()
 

# ============================================
# 19. QUIVER PLOT (VECTOR FIELD)
# ============================================
def quiver_plot():
    x = np.arange(0, 10, 1)
    y = np.arange(0, 10, 1)
    X, Y = np.meshgrid(x, y)
    U = np.cos(X)
    V = np.sin(Y)
    
    plt.figure(figsize=(10, 8))
    plt.quiver(X, Y, U, V, alpha=0.8)
    plt.title('Quiver Plot - Vector Field')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, alpha=0.3)
    plt.show()
 
 
# ============================================
# 20. SUBPLOT EXAMPLE
# ============================================
def subplot_example():
    x = np.linspace(0, 10, 100)
    
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Line
    plt.subplot(2, 3, 1)
    plt.plot(x, np.sin(x))
    plt.title('Sine Wave')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Bar
    plt.subplot(2, 3, 2)
    plt.bar(['A', 'B', 'C'], [3, 7, 5])
    plt.title('Bar Chart')
    
    # Plot 3: Scatter
    plt.subplot(2, 3, 3)
    plt.scatter(x, np.cos(x))
    plt.title('Scatter Plot')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Histogram
    plt.subplot(2, 3, 4)
    plt.hist(np.random.randn(1000), bins=30)
    plt.title('Histogram')
    
    # Plot 5: Pie
    plt.subplot(2, 3, 5)
    plt.pie([30, 25, 20, 25], labels=['A', 'B', 'C', 'D'], autopct='%1.1f%%')
    plt.title('Pie Chart')
    
    # Plot 6: Box
    plt.subplot(2, 3, 6)
    plt.boxplot([np.random.randn(100), np.random.randn(100)])
    plt.title('Box Plot')
    
    plt.tight_layout()
    plt.show()
 
 
# ============================================
# MAIN FUNCTION - RUN ALL PLOTS
# ============================================
def main():
    print("Generating all plot types...")
    print("\n1. Line Plot")
    line_plot()
    
    print("\n2. Bar Chart")
    bar_chart()
    
    print("\n3. Horizontal Bar Chart")
    horizontal_bar()
    
    print("\n4. Scatter Plot")
    scatter_plot()
    
    print("\n5. Pie Chart")
    pie_chart()
    
    print("\n6. Histogram")
    histogram()
    
    print("\n7. Box Plot")
    box_plot()
    
    print("\n8. Area Plot")
    area_plot()
    
    print("\n9. Stacked Bar Chart")
    stacked_bar()
    
    print("\n10. Grouped Bar Chart")
    grouped_bar()
    
    print("\n11. Heatmap")
    heatmap()
    
    print("\n12. Contour Plot")
    contour_plot()
    
    print("\n13. 3D Surface Plot")
    surface_3d()
    
    print("\n14. Violin Plot")
    violin_plot()
    
    print("\n15. Step Plot")
    step_plot()
    
    print("\n16. Error Bar Plot")
    error_bar()
    
    print("\n17. Polar Plot")
    polar_plot()
    
    print("\n18. Stem Plot")
    stem_plot()
    
    print("\n19. Quiver Plot")
    quiver_plot()
    
    print("\n20. Subplot Example")
    subplot_example()
    
    print("\n✅ All plots generated successfully!")
 
 
if __name__ == "__main__":
    main()
