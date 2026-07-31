print("✅ NEW SIDEBAR.PY IS LOADED")
import streamlit as st


def render_sidebar_drawer(
    metadata,
    decision_threshold,
    load_example_customer,
    load_low_risk_customer,
    reset_customer,
):
    """
    Modern sidebar for customer churn prediction.
    """

    # -------------------------------------------------------
    # Header
    # -------------------------------------------------------

    st.sidebar.markdown(
        """
        <div style="padding-bottom:0.8rem">
            <h2 style="margin-bottom:0;">
                📊 Customer Filters
            </h2>
            <p style="font-size:0.82rem;color:#9ca3af;">
                Configure customer attributes before prediction.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------
    # Quick Actions
    # -------------------------------------------------------

    col1, col2 = st.sidebar.columns(2)

    with col1:
        st.button(
            "High Risk",
            key="btn_high",
            use_container_width=True,
            on_click=load_example_customer,
        )

    with col2:
        st.button(
            "Low Risk",
            key="btn_low",
            use_container_width=True,
            on_click=load_low_risk_customer,
        )

    st.sidebar.button(
        "Reset",
        key="btn_reset",
        use_container_width=True,
        on_click=reset_customer,
    )

    st.sidebar.divider()

    # -------------------------------------------------------
    # Form
    # -------------------------------------------------------

    with st.sidebar.form("customer_form"):

        # ===================================================
        # Customer Details
        # ===================================================

        with st.expander("📋 Customer Details", expanded=True):

            st.number_input(
                "Tenure (Months)",
                min_value=0,
                max_value=100,
                key="tenure",
            )

            st.selectbox(
                "Contract",
                [
                    "Month-to-month",
                    "One year",
                    "Two year",
                ],
                key="contract",
            )

        # ===================================================
        # Demographics
        # ===================================================

        with st.expander("👤 Demographics"):

            st.radio(
                "Gender",
                [
                    "Female",
                    "Male",
                ],
                horizontal=True,
                key="gender",
            )

            st.radio(
                "Senior Citizen",
                [
                    "No",
                    "Yes",
                ],
                horizontal=True,
                key="senior_citizen",
            )

            st.radio(
                "Partner",
                [
                    "No",
                    "Yes",
                ],
                horizontal=True,
                key="partner",
            )

            st.radio(
                "Dependents",
                [
                    "No",
                    "Yes",
                ],
                horizontal=True,
                key="dependents",
            )
        # ===================================================
        # Billing
        # ===================================================

        with st.expander("💳 Billing"):

            st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                step=0.5,
                key="monthly_charges",
            )

            st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                step=1.0,
                key="total_charges",
            )

            st.radio(
                "Paperless Billing",
                ["No", "Yes"],
                horizontal=True,
                key="paperless_billing",
            )

            st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                key="payment_method",
            )

        # ===================================================
        # Services
        # ===================================================

        with st.expander("📡 Services"):

            st.radio(
                "Phone Service",
                ["No", "Yes"],
                horizontal=True,
                key="phone_service",
            )

            st.selectbox(
                "Multiple Lines",
                [
                    "No",
                    "Yes",
                    "No phone service",
                ],
                key="multiple_lines",
            )

            st.selectbox(
                "Internet Service",
                [
                    "DSL",
                    "Fiber optic",
                    "No",
                ],
                key="internet_service",
            )

            st.selectbox(
                "Online Security",
                [
                    "No",
                    "Yes",
                    "No internet service",
                ],
                key="online_security",
            )

            st.selectbox(
                "Online Backup",
                [
                    "No",
                    "Yes",
                    "No internet service",
                ],
                key="online_backup",
            )

            st.selectbox(
                "Device Protection",
                [
                    "No",
                    "Yes",
                    "No internet service",
                ],
                key="device_protection",
            )

            st.selectbox(
                "Tech Support",
                [
                    "No",
                    "Yes",
                    "No internet service",
                ],
                key="tech_support",
            )

            st.selectbox(
                "Streaming TV",
                [
                    "No",
                    "Yes",
                    "No internet service",
                ],
                key="streaming_tv",
            )

            st.selectbox(
                "Streaming Movies",
                [
                    "No",
                    "Yes",
                    "No internet service",
                ],
                key="streaming_movies",
            )

            submitted = st.form_submit_button(
                "🚀 Run Risk Prediction",
                type="primary",
                use_container_width=True,
            )
    # ===================================================
    # Model Governance
    # ===================================================

    st.sidebar.divider()

    with st.sidebar.expander("📈 Model Governance", expanded=False):

        metrics = metadata.get("metrics", {})

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Accuracy",
                f"{metrics.get('Accuracy', 0):.3f}",
            )

            st.metric(
                "Recall",
                f"{metrics.get('Recall', 0):.3f}",
            )

        with c2:
            st.metric(
                "Precision",
                f"{metrics.get('Precision', 0):.3f}",
            )

            st.metric(
                "ROC AUC",
                f"{metrics.get('ROC_AUC', 0):.3f}",
            )

        st.caption("Model")

        st.code(
            metadata.get(
                "model_type",
                "LogisticRegression",
            ),
            language="text",
        )

        st.caption("Decision Threshold")

        st.progress(float(decision_threshold))

        st.write(f"Current Threshold: **{decision_threshold:.2f}**")

    return submitted
