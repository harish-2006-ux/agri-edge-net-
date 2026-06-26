# PART 2: Analysis, Model Info, and other pages

# ============================================================================
# PAGE: ANALYSIS
# ============================================================================

def page_analysis():
    """Advanced analysis page with image upload and predictions."""
    load_custom_css()
    
    st.markdown("""
    <div class="main-header">
        <h1>🔬 Disease Analysis & Prediction</h1>
        <p>Upload crop images and weather data for multi-task predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two main sections
    tab1, tab2, tab3 = st.tabs(["📸 Image Analysis", "📈 Batch Processing", "💰 Economic Simulation"])
    
    with tab1:
        analysis_single_image()
    
    with tab2:
        analysis_batch()
    
    with tab3:
        economic_simulator()


def analysis_single_image():
    """Single image analysis interface."""
    col_upload, col_weather = st.columns([1, 1])
    
    with col_upload:
        st.markdown("### 📸 Upload Crop Image")
        uploaded_file = st.file_uploader(
            "Choose a leaf image (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of a crop leaf for disease diagnosis"
        )
        
        if uploaded_file:
            img_tensor, processed_img, orig_size = preprocess_image(uploaded_file)
            
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.image(Image.open(uploaded_file), caption=f"Original ({orig_size[0]}×{orig_size[1]})", use_container_width=True)
            with col_img2:
                st.image(processed_img, caption="Processed (224×224)", use_container_width=True)
            
            st.session_state['current_image'] = img_tensor
            st.session_state['original_image'] = Image.open(uploaded_file)
