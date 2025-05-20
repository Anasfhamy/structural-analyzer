
import streamlit as st
import numpy as np
import pandas as pd

class StructuralAnalyzer:
    def __init__(self, E, I, L):
        self.E = E  # Elastic modulus
        self.I = I  # Moment of inertia
        self.L = L  # Length
        self.k = (E * I) / (L**3)  # Stiffness coefficient
        self.K = self._create_stiffness_matrix()

    def _create_stiffness_matrix(self):
        k = self.k
        L = self.L
        return np.array([
            [k * 12,   k * 6*L,    -k * 12,    0,         0,          -k * 6*L],
            [k * 6*L,  k * 4*L**2,  0,         0,         -k * 6*L,   -k * 4*L**2],
            [-k * 12,  0,           k * 12,    -k * 6*L,  0,          k * 6*L],
            [0,        0,          -k * 6*L,   k * 4*L**2, k * 6*L,   -k * 4*L**2],
            [0,       -k * 6*L,     0,         k * 6*L,   k * 12,     0],
            [-k * 6*L, -k * 4*L**2, k * 6*L,  -k * 4*L**2, 0,         k * 8*L**2]
        ])

    def calculate_displacements(self, F, fixed_dofs=None):
        if fixed_dofs is None:
            fixed_dofs = [0, 1, 2, 3, 4]
        n = len(self.K)
        free_dofs = list(set(range(n)) - set(fixed_dofs))
        K_reduced = self.K[np.ix_(free_dofs, free_dofs)]
        F_reduced = F[free_dofs]
        d = np.zeros(n)
        d[free_dofs] = np.linalg.solve(K_reduced, F_reduced)
        return d

    def calculate_reactions(self, d, F):
        return np.dot(self.K, d) - F

    def calculate_internal_forces(self, d):
        k = self.k
        L = self.L
        force_transformation = np.array([
            [-k * 12,  -k * 6*L,   k * 12,   k * 6*L],
            [-k * 6*L, -k * 4*L**2, k * 6*L,  k * 2*L**2],
            [k * 12,   k * 6*L,   -k * 12,  -k * 6*L],
            [k * 6*L,  k * 2*L**2, -k * 6*L, -k * 4*L**2]
        ])
        return np.dot(force_transformation, d[[0, 1, 2, 3]])

    def get_deflection_moment_plot_data(self, d, num_points=20):
        L = self.L
        x_vals = np.linspace(0, L, num_points)

        V1, theta1, V2, theta2 = d[0], d[1], d[2], d[3]

        def shape_function_deflection(x):
            xi = x / L
            return (1 - 3*xi**2 + 2*xi**3)*V1 + L*(xi - 2*xi**2 + xi**3)*theta1 +                    (3*xi**2 - 2*xi**3)*V2 + L*(-xi**2 + xi**3)*theta2

        def shape_function_moment(x):
            xi = x / L
            d2w_dx2 = (-6/L**2 + 6*xi/L**2)*V1 + (1 - 4*xi + 3*xi**2)*theta1 +                       (6/L**2 - 6*xi/L**2)*V2 + (-2 + 3*xi)*theta2
            return self.E * self.I * d2w_dx2

        deflections = [shape_function_deflection(xi) for xi in x_vals]
        moments = [shape_function_moment(xi) for xi in x_vals]

        return x_vals, deflections, moments

def main():
    st.set_page_config(
        page_title="Structural Analysis Tool",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    st.markdown("""
        <style>
        .stApp { max-width: 100%; padding: 1rem; }
        .st-emotion-cache-1r6slb0, .st-emotion-cache-1d3w5wq {
            padding: 1.5rem !important;
        }
        @media (max-width: 768px) {
            .st-emotion-cache-1r6slb0, .st-emotion-cache-1d3w5wq {
                padding: 0.5rem !important;
            }
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("📐 برنامج التحليل الإنشائي")

    st.sidebar.markdown("## 🔢 إدخال البيانات")

    E = st.sidebar.number_input("معامل المرونة E (N/m²)", value=210e9, format="%e")
    I = st.sidebar.number_input("عزم القصور الذاتي I (m⁴)", value=0.0002, format="%e")
    L = st.sidebar.number_input("الطول L (m)", value=3.0, min_value=0.1)

    st.sidebar.markdown("### 🔒 اختر درجات الحرية المقيدة")
    fixed_dofs = st.sidebar.multiselect(
        "اختر DOFs التي تم تقييدها",
        options=[0, 1, 2, 3, 4, 5],
        default=[0, 1, 2, 3, 4]
    )

    st.sidebar.markdown("### 🎯 إدخال القوى على درجات الحرية (F1 → F6)")
    force_vector = []
    for i in range(6):
        force_value = st.sidebar.number_input(f"F{i+1} (N أو N·m)", value=0.0, format="%.2f")
        force_vector.append(force_value)
    force_vector = np.array(force_vector)

    analyzer = StructuralAnalyzer(E, I, L)

    try:
        d = analyzer.calculate_displacements(force_vector, fixed_dofs=fixed_dofs)
    except np.linalg.LinAlgError:
        st.error("❌ النظام غير قابل للحل. تأكد من إدخال قيود كافية.")
        st.stop()

    R = analyzer.calculate_reactions(d, force_vector)
    internal_forces = analyzer.calculate_internal_forces(d)
    x_vals, deflections, moments = analyzer.get_deflection_moment_plot_data(d)

    tabs = st.tabs([
        "Stiffness Matrix", 
        "Displacements", 
        "Reactions", 
        "Internal Forces", 
        "📈 الرسوم البيانية"
    ])

    with tabs[0]:
        st.subheader("Stiffness Matrix")
        st.dataframe(analyzer.K, use_container_width=True)

    with tabs[1]:
        st.subheader("Displacements (m, rad)")
        st.dataframe(pd.DataFrame(d, columns=['Value']), use_container_width=True)

    with tabs[2]:
        st.subheader("Reactions (N, N⋅m)")
        st.dataframe(pd.DataFrame(R, columns=['Value']), use_container_width=True)

    with tabs[3]:
        st.subheader("Internal Forces (N, N⋅m)")
        st.dataframe(pd.DataFrame(internal_forces, columns=['Value']), use_container_width=True)

    with tabs[4]:
        st.subheader("📉 شكل الانفعالات (Deflection Curve)")
        st.line_chart(pd.DataFrame({"x (m)": x_vals, "Deflection (m)": deflections}).set_index("x (m)"))

        st.subheader("📊 توزيع العزوم (Moment Diagram)")
        st.line_chart(pd.DataFrame({"x (m)": x_vals, "Moment (N⋅m)": moments}).set_index("x (m)"))

if __name__ == "__main__":
    main()
