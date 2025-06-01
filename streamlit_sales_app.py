# app.py - Sales Automation System (Debugged Version)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import pickle
import sqlite3
from datetime import datetime
import io
import base64
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    st.warning("psutil not installed. Memory monitoring disabled.")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    st.error("OpenAI package not installed. Please install with: pip install openai")

# Configure page
st.set_page_config(
    page_title="Sales Automation System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state with better defaults
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'config': {},
        'model': None,
        'data_processed': False,
        'model_trained': False,
        'sales_system': None,
        'crm_data': None,
        'leads_data': None,
        'processing_results': [],
        'model_metrics': {}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

class SalesAutomationSystem:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.required_columns = ['company_name', 'industry', 'won']
        
    def validate_crm_data(self, df):
        """Validate that required columns exist"""
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        return True
        
    def process_crm_data(self, df):
        """Process uploaded CRM data with better error handling"""
        try:
            # Validate required columns
            self.validate_crm_data(df)
            
            # Create a copy to avoid modifying original
            df = df.copy()
            
            # Basic data cleaning
            df = df.dropna(subset=self.required_columns)
            
            # Handle boolean conversion more robustly
            if df['won'].dtype == 'object':
                df['won'] = df['won'].astype(str).str.lower().isin(['true', '1', 'yes', 'won'])
            else:
                df['won'] = df['won'].astype(bool)
            
            # Feature engineering with error handling
            numeric_cols = ['deal_value', 'time_to_close']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Handle missing categorical columns
            categorical_cols = ['industry', 'company_size', 'location']
            for col in categorical_cols:
                if col in df.columns:
                    df[col] = df[col].fillna('Unknown')
            
            return df
            
        except Exception as e:
            st.error(f"Error processing CRM data: {str(e)}")
            return None
    
    def prepare_features(self, df, is_training=True):
        """Prepare features for model training/prediction"""
        try:
            feature_cols = ['industry', 'company_size', 'deal_value', 'time_to_close', 'location']
            available_cols = [col for col in feature_cols if col in df.columns]
            
            if not available_cols:
                raise ValueError("No valid feature columns found")
            
            X = df[available_cols].copy()
            
            # Handle categorical variables
            categorical_cols = ['industry', 'company_size', 'location']
            for col in categorical_cols:
                if col in X.columns:
                    if is_training:
                        if col not in self.label_encoders:
                            self.label_encoders[col] = LabelEncoder()
                        X[col] = self.label_encoders[col].fit_transform(X[col].fillna('Unknown'))
                    else:
                        if col in self.label_encoders:
                            # Handle unseen categories
                            unique_values = X[col].fillna('Unknown').unique()
                            known_values = self.label_encoders[col].classes_
                            
                            # Map unknown values to a default
                            X[col] = X[col].fillna('Unknown')
                            X[col] = X[col].apply(lambda x: x if x in known_values else 'Unknown')
                            X[col] = self.label_encoders[col].transform(X[col])
                        else:
                            X[col] = 0  # Default value if encoder not available
            
            # Handle numeric variables
            numeric_cols = ['deal_value', 'time_to_close']
            for col in numeric_cols:
                if col in X.columns:
                    X[col] = X[col].fillna(0)
            
            if is_training:
                self.feature_columns = X.columns.tolist()
            
            return X
            
        except Exception as e:
            st.error(f"Error preparing features: {str(e)}")
            return None
    
    def train_model(self, crm_df):
        """Train the prediction model with better error handling"""
        try:
            if crm_df is None or len(crm_df) == 0:
                raise ValueError("No valid training data provided")
            
            X = self.prepare_features(crm_df, is_training=True)
            if X is None:
                return None
                
            y = crm_df['won'].astype(int)
            
            if len(X) < 10:
                raise ValueError("Not enough data for training (minimum 10 samples required)")
            
            # Check for class balance
            if y.sum() == 0 or y.sum() == len(y):
                st.warning("All samples have the same class. Model may not be reliable.")
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data with stratification if possible
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42, stratify=y
                )
            except ValueError:
                # Fall back to non-stratified split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )
            
            # Train model with better parameters
            self.model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                min_samples_split=5,
                min_samples_leaf=2
            )
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            
            # Handle case where model only predicts one class
            try:
                y_pred_proba = self.model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_pred_proba)
            except (ValueError, IndexError):
                auc = 0.5  # Random performance
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'auc': auc,
                'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            }
            
            return metrics
            
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return None
    
    def predict_prospect(self, prospect_data):
        """Predict conversion probability for a prospect"""
        try:
            if not self.model:
                return 0.0
            
            # Prepare features
            features = {}
            for col in self.feature_columns:
                if col in ['industry', 'company_size', 'location']:
                    value = prospect_data.get(col, 'Unknown')
                    if col in self.label_encoders:
                        try:
                            # Check if value is in known classes
                            if value in self.label_encoders[col].classes_:
                                features[col] = self.label_encoders[col].transform([value])[0]
                            else:
                                # Use 'Unknown' as fallback
                                if 'Unknown' in self.label_encoders[col].classes_:
                                    features[col] = self.label_encoders[col].transform(['Unknown'])[0]
                                else:
                                    features[col] = 0
                        except (ValueError, KeyError):
                            features[col] = 0
                    else:
                        features[col] = 0
                else:
                    features[col] = float(prospect_data.get(col, 0))
            
            # Create DataFrame and predict
            feature_df = pd.DataFrame([features])
            
            # Ensure all required columns are present
            for col in self.feature_columns:
                if col not in feature_df.columns:
                    feature_df[col] = 0
            
            feature_df = feature_df[self.feature_columns]
            feature_scaled = self.scaler.transform(feature_df)
            
            try:
                probability = self.model.predict_proba(feature_scaled)[0][1]
            except IndexError:
                # Handle case where model only has one class
                probability = 0.5
            
            return float(probability)
            
        except Exception as e:
            st.error(f"Error predicting prospect: {str(e)}")
            return 0.0

def show_memory_usage():
    """Display current memory usage"""
    if not PSUTIL_AVAILABLE:
        return
        
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        st.sidebar.metric("Memory Usage", f"{memory_mb:.1f} MB")
        
        if memory_mb > 400:
            st.sidebar.warning("High memory usage detected!")
            if st.sidebar.button("Clear Cache"):
                st.cache_data.clear()
                try:
                    st.cache_resource.clear()
                except AttributeError:
                    pass  # Older Streamlit versions
                st.rerun()
    except Exception as e:
        st.sidebar.info(f"Memory monitoring error: {str(e)}")

def research_company(company_name, domain, openai_key):
    """Research company using OpenAI with better error handling"""
    if not OPENAI_AVAILABLE:
        return {"error": "OpenAI package not installed"}
    
    if not openai_key:
        return {"error": "OpenAI API key required"}
    
    try:
        client = openai.OpenAI(api_key=openai_key)
        
        prompt = f"""
        Research the company "{company_name}" (website: {domain}) and provide:
        1. Recent news and developments
        2. Business challenges they might face
        3. Key growth indicators
        4. Industry trends affecting them
        
        Provide a brief summary in 2-3 sentences.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a business research analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.3,
            timeout=30  # Add timeout
        )
        
        return {"research": response.choices[0].message.content}
        
    except openai.RateLimitError:
        return {"error": "OpenAI rate limit exceeded. Please try again later."}
    except openai.AuthenticationError:
        return {"error": "Invalid OpenAI API key"}
    except Exception as e:
        return {"error": f"Research failed: {str(e)}"}

def generate_email(prospect_data, research_data, openai_key):
    """Generate personalized email with better error handling"""
    if not OPENAI_AVAILABLE:
        return {"error": "OpenAI package not installed"}
    
    if not openai_key:
        return {"error": "OpenAI API key required"}
    
    try:
        client = openai.OpenAI(api_key=openai_key)
        
        research_text = research_data.get('research', 'No research available')
        
        prompt = f"""
        Write a personalized cold email for {prospect_data['company_name']}:
        
        Company: {prospect_data['company_name']}
        Industry: {prospect_data.get('industry', 'Unknown')}
        Research: {research_text}
        
        Guidelines:
        1. Keep under 100 words
        2. Reference specific company insights
        3. Professional but conversational tone
        4. Clear call-to-action
        5. Include compelling subject line
        
        Format:
        Subject: [subject line]
        
        [email body]
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert sales copywriter."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7,
            timeout=30  # Add timeout
        )
        
        email_content = response.choices[0].message.content
        
        # Parse subject and body with better error handling
        try:
            lines = email_content.split('\n')
            subject_line = next((line for line in lines if line.lower().startswith('subject:')), lines[0] if lines else 'Partnership Opportunity')
            subject = subject_line.replace('Subject:', '').replace('subject:', '').strip()
            
            # Find email body (everything after subject line)
            body_start = next((i for i, line in enumerate(lines) if line.lower().startswith('subject:')), -1) + 2
            if body_start < len(lines):
                body = '\n'.join(lines[body_start:]).strip()
            else:
                body = email_content
                
        except Exception:
            subject = 'Partnership Opportunity'
            body = email_content
        
        return {"subject": subject, "body": body}
        
    except openai.RateLimitError:
        return {"error": "OpenAI rate limit exceeded. Please try again later."}
    except openai.AuthenticationError:
        return {"error": "Invalid OpenAI API key"}
    except Exception as e:
        return {"error": f"Email generation failed: {str(e)}"}

def main():
    # Initialize session state
    initialize_session_state()
    
    st.title("🎯 Sales Automation System")
    st.markdown("**AI-Powered Lead Scoring, Research & Email Generation**")
    
    # Show memory usage
    show_memory_usage()
    
    # Initialize system
    if st.session_state.sales_system is None:
        st.session_state.sales_system = SalesAutomationSystem()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Try to get API key from secrets first
    openai_key = ""
    try:
        openai_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        pass
    
    if not openai_key:
        openai_key = st.sidebar.text_input(
            "OpenAI API Key", 
            type="password",
            help="Required for company research and email generation"
        )
    else:
        st.sidebar.success("✅ API Key loaded from secrets")
    
    min_score = st.sidebar.slider(
        "Minimum Conversion Score", 
        0.0, 1.0, 0.7, 0.1,
        help="Only process prospects above this score"
    )
    
    # Clear cache button
    if st.sidebar.button("🗑️ Clear Cache"):
        st.cache_data.clear()
        try:
            st.cache_resource.clear()
        except AttributeError:
            pass
        # Clear session state
        for key in list(st.session_state.keys()):
            if key not in ['config']:
                del st.session_state[key]
        st.rerun()
    
    st.session_state.config = {
        'openai_key': openai_key,
        'min_score': min_score
    }
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 Data Import", 
        "🤖 Model Training", 
        "⚡ Process Leads", 
        "📧 Results",
        "📊 Analytics"
    ])
    
    with tab1:
        st.header("📁 Data Import")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("CRM Data")
            st.info("Required columns: company_name, industry, won")
            st.info("Optional columns: company_size, deal_value, time_to_close, location")
            
            crm_file = st.file_uploader(
                "Upload CRM CSV", 
                type=['csv'],
                key="crm_upload"
            )
            
            if crm_file:
                try:
                    crm_df = pd.read_csv(crm_file)
                    st.write("**Original columns:**", list(crm_df.columns))
                    
                    processed_df = st.session_state.sales_system.process_crm_data(crm_df)
                    
                    if processed_df is not None:
                        st.session_state.crm_data = processed_df
                        
                        st.success(f"✅ Loaded {len(processed_df)} CRM records")
                        st.dataframe(processed_df.head())
                        
                        # Data summary
                        won_rate = (processed_df['won'].sum() / len(processed_df)) * 100
                        st.metric("Conversion Rate", f"{won_rate:.1f}%")
                        
                        # Show data info
                        st.write("**Data Info:**")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write("- Total records:", len(processed_df))
                            st.write("- Won deals:", processed_df['won'].sum())
                        with col_b:
                            st.write("- Lost deals:", (~processed_df['won']).sum())
                            st.write("- Industries:", processed_df['industry'].nunique())
                    
                except Exception as e:
                    st.error(f"❌ Error loading CRM data: {e}")
        
        with col2:
            st.subheader("Lead Forensics Data")
            st.info("Expected columns: company_name, domain, industry, company_size")
            st.info("Optional columns: pages_visited, session_duration, return_visits, location")
            
            leads_file = st.file_uploader(
                "Upload Leads CSV", 
                type=['csv'],
                key="leads_upload"
            )
            
            if leads_file:
                try:
                    leads_df = pd.read_csv(leads_file)
                    st.write("**Columns found:**", list(leads_df.columns))
                    
                    # Basic validation
                    required_lead_cols = ['company_name']
                    missing_cols = [col for col in required_lead_cols if col not in leads_df.columns]
                    
                    if missing_cols:
                        st.error(f"❌ Missing required columns: {missing_cols}")
                    else:
                        st.session_state.leads_data = leads_df
                        
                        st.success(f"✅ Loaded {len(leads_df)} lead records")
                        st.dataframe(leads_df.head())
                        
                        # Show lead info
                        st.write("**Lead Info:**")
                        if 'industry' in leads_df.columns:
                            st.write("- Industries:", leads_df['industry'].nunique())
                        if 'company_size' in leads_df.columns:
                            st.write("- Company sizes:", leads_df['company_size'].nunique())
                    
                except Exception as e:
                    st.error(f"❌ Error loading leads data: {e}")
    
    with tab2:
        st.header("🤖 Model Training")
        
        if st.session_state.crm_data is None:
            st.warning("⚠️ Please upload CRM data first")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Training data:** {len(st.session_state.crm_data)} records")
                
                if st.button("🚀 Train Model", type="primary"):
                    with st.spinner("Training model..."):
                        try:
                            metrics = st.session_state.sales_system.train_model(st.session_state.crm_data)
                            
                            if metrics:
                                st.session_state.model_trained = True
                                st.session_state.model_metrics = metrics
                                
                                st.success("✅ Model trained successfully!")
                                
                                # Display metrics
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                                with col_b:
                                    st.metric("AUC Score", f"{metrics['auc']:.3f}")
                                with col_c:
                                    try:
                                        precision = metrics['classification_report']['weighted avg']['precision']
                                        st.metric("Precision", f"{precision:.3f}")
                                    except KeyError:
                                        st.metric("Precision", "N/A")
                            else:
                                st.error("❌ Model training failed")
                                
                        except Exception as e:
                            st.error(f"❌ Training failed: {e}")
            
            with col2:
                if st.session_state.model_trained:
                    st.success("✅ Model Ready")
                    if st.session_state.model_metrics:
                        st.write("**Model Performance:**")
                        st.write(f"- Accuracy: {st.session_state.model_metrics.get('accuracy', 0):.3f}")
                        st.write(f"- AUC: {st.session_state.model_metrics.get('auc', 0):.3f}")
                else:
                    st.info("ℹ️ Model Not Trained")
    
    with tab3:
        st.header("⚡ Process New Leads")
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train the model first")
        elif st.session_state.leads_data is None:
            st.warning("⚠️ Please upload leads data first")
        else:
            batch_size = st.slider("Batch Size", 1, min(50, len(st.session_state.leads_data)), 10)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"Will process {batch_size} leads from {len(st.session_state.leads_data)} total")
            with col2:
                if not openai_key:
                    st.warning("⚠️ No OpenAI key - research/email disabled")
            
            if st.button("⚡ Process Leads", type="primary"):
                leads_df = st.session_state.leads_data.head(batch_size)
                results = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, (_, lead) in enumerate(leads_df.iterrows()):
                    progress_bar.progress((idx + 1) / len(leads_df))
                    status_text.text(f"Processing {lead['company_name']}...")
                    
                    # Predict conversion probability
                    prospect_data = {
                        'company_name': lead['company_name'],
                        'industry': lead.get('industry', 'Unknown'),
                        'company_size': lead.get('company_size', 'Unknown'),
                        'location': lead.get('location', 'Unknown'),
                        'deal_value': 0,
                        'time_to_close': 0
                    }
                    
                    score = st.session_state.sales_system.predict_prospect(prospect_data)
                    
                    if score >= min_score:
                        # Research company (only if API key available)
                        if openai_key:
                            research = research_company(
                                lead['company_name'], 
                                lead.get('domain', ''), 
                                openai_key
                            )
                            
                            # Generate email
                            email = generate_email(prospect_data, research, openai_key)
                        else:
                            research = {"research": "Research disabled - no API key"}
                            email = {"subject": "Partnership Opportunity", "body": "Email generation disabled - no API key"}
                        
                        results.append({
                            'company_name': lead['company_name'],
                            'industry': lead.get('industry', 'Unknown'),
                            'score': score,
                            'research': research.get('research', research.get('error', 'No research available')),
                            'email_subject': email.get('subject', 'Partnership Opportunity'),
                            'email_body': email.get('body', email.get('error', 'Email generation failed')),
                            'status': 'pending'
                        })
                
                st.session_state.processing_results = results
                progress_bar.progress(1.0)
                status_text.text("✅ Processing complete!")
                
                st.success(f"🎉 Processed {len(leads_df)} leads, found {len(results)} qualified prospects")
    
    with tab4:
        st.header("📧 Generated Emails")
        
        if not st.session_state.processing_results:
            st.info("ℹ️ No results yet. Process some leads first.")
        else:
            results = st.session_state.processing_results
            
            # Add filter options
            col1, col2 = st.columns(2)
            with col1:
                min_display_score = st.slider("Minimum Score to Display", 0.0, 1.0, 0.0, 0.1)
            with col2:
                status_filter = st.selectbox("Status Filter", ["All", "pending", "approved", "rejected"])
            
            # Filter results
            filtered_results = results
            if min_display_score > 0:
                filtered_results = [r for r in filtered_results if r['score'] >= min_display_score]
            if status_filter != "All":
                filtered_results = [r for r in filtered_results if r['status'] == status_filter]
            
            st.write(f"Showing {len(filtered_results)} of {len(results)} results")
            
            for idx, result in enumerate(filtered_results):
                # Find original index
                orig_idx = next(i for i, r in enumerate(results) if r == result)
                
                with st.expander(f"{result['company_name']} - Score: {result['score']:.2f} - {result['status'].title()}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("📧 Generated Email")
                        st.write(f"**Subject:** {result['email_subject']}")
                        st.text_area("Email Body", result['email_body'], height=100, key=f"email_{orig_idx}")
                        
                        st.subheader("🔍 Company Research")
                        st.write(result['research'])
                    
                    with col2:
                        st.metric("Conversion Score", f"{result['score']:.1%}")
                        st.metric("Industry", result['industry'])
                        st.metric("Status", result['status'].title())
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button("✅ Approve", key=f"approve_{orig_idx}"):
                                st.session_state.processing_results[orig_idx]['status'] = 'approved'
                                st.success("Email approved!")
                                st.rerun()
                        
                        with col_b:
                            if st.button("❌ Reject", key=f"reject_{orig_idx}"):
                                st.session_state.processing_results[orig_idx]['status'] = 'rejected'
                                st.success("Email rejected!")
                                st.rerun()
                        
                        if st.button("🔄 Regenerate", key=f"regen_{orig_idx}"):
                            if openai_key:
                                with st.spinner("Regenerating..."):
                                    new_email = generate_email(
                                        {'company_name': result['company_name'], 'industry': result['industry']},
                                        {'research': result['research']},
                                        openai_key
                                    )
                                    st.session_state.processing_results[orig_idx]['email_subject'] = new_email.get('subject', 'Partnership Opportunity')
                                    st.session_state.processing_results[orig_idx]['email_body'] = new_email.get('body', 'Email generation failed')
                                    st.rerun()
                            else:
                                st.error("OpenAI API key required for regeneration")
            
            # Export functionality
            if st.button("📥 Export Results"):
                try:
                    df_results = pd.DataFrame(results)
                    csv = df_results.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="sales_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv">Download CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    st.success("✅ Export ready!")
                except Exception as e:
                    st.error(f"❌ Export failed: {e}")
    
    with tab5:
        st.header("📊 Analytics Dashboard")
        
        if st.session_state.processing_results:
            results = st.session_state.processing_results
            
            if results:
                # Score distribution
                scores = [r['score'] for r in results]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Prospects", len(results))
                with col2:
                    st.metric("Avg Score", f"{np.mean(scores):.1%}")
                with col3:
                    approved_count = len([r for r in results if r.get('status') == 'approved'])
                    st.metric("Approved Emails", approved_count)
                with col4:
                    high_score_count = len([r for r in results if r['score'] >= 0.8])
                    st.metric("High Score (≥80%)", high_score_count)
                
                # Score histogram
                st.subheader("📈 Score Distribution")
                fig_data = pd.DataFrame({'Score': scores})
                st.bar_chart(fig_data['Score'])
                
                # Industry breakdown
                industries = [r['industry'] for r in results]
                industry_counts = pd.Series(industries).value_counts()
                
                st.subheader("🏭 Industry Breakdown")
                st.bar_chart(industry_counts)
                
                # Status breakdown
                statuses = [r.get('status', 'pending') for r in results]
                status_counts = pd.Series(statuses).value_counts()
                
                st.subheader("📋 Status Breakdown")
                st.bar_chart(status_counts)
        else:
            st.info("ℹ️ No analytics data available yet.")

if __name__ == "__main__":
    main()
