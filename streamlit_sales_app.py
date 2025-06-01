# app.py - Main Streamlit application
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import openai
import pickle
import sqlite3
from datetime import datetime
import io
import base64

# Configure page
st.set_page_config(
    page_title="Sales Automation System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'config' not in st.session_state:
    st.session_state.config = {}
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

class SalesAutomationSystem:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        
    def process_crm_data(self, df):
        """Process uploaded CRM data"""
        # Basic data cleaning
        df = df.dropna(subset=['company_name', 'industry', 'won'])
        df['won'] = df['won'].astype(bool)
        
        # Feature engineering
        numeric_cols = ['deal_value', 'time_to_close']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for model training"""
        feature_cols = ['industry', 'company_size', 'deal_value', 'time_to_close', 'location']
        available_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[available_cols].copy()
        
        # Handle categorical variables
        categorical_cols = ['industry', 'company_size', 'location']
        for col in categorical_cols:
            if col in X.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    X[col] = self.label_encoders[col].fit_transform(X[col].fillna('unknown'))
                else:
                    X[col] = self.label_encoders[col].transform(X[col].fillna('unknown'))
        
        # Handle numeric variables
        numeric_cols = ['deal_value', 'time_to_close']
        for col in numeric_cols:
            if col in X.columns:
                X[col] = X[col].fillna(0)
        
        self.feature_columns = X.columns.tolist()
        return X
    
    def train_model(self, crm_df):
        """Train the prediction model"""
        X = self.prepare_features(crm_df)
        y = crm_df['won'].astype(int)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        return metrics
    
    def predict_prospect(self, prospect_data):
        """Predict conversion probability for a prospect"""
        if not self.model:
            return 0.0
        
        # Prepare features
        features = {}
        for col in self.feature_columns:
            if col in ['industry', 'company_size', 'location']:
                value = prospect_data.get(col, 'unknown')
                if col in self.label_encoders:
                    try:
                        features[col] = self.label_encoders[col].transform([value])[0]
                    except ValueError:
                        features[col] = 0
                else:
                    features[col] = 0
            else:
                features[col] = prospect_data.get(col, 0)
        
        # Create DataFrame and predict
        feature_df = pd.DataFrame([features])
        feature_scaled = self.scaler.transform(feature_df[self.feature_columns])
        probability = self.model.predict_proba(feature_scaled)[0][1]
        
        return probability

def research_company(company_name, domain, openai_key):
    """Research company using OpenAI"""
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
            temperature=0.3
        )
        
        return {"research": response.choices[0].message.content}
        
    except Exception as e:
        return {"error": f"Research failed: {str(e)}"}

def generate_email(prospect_data, research_data, openai_key):
    """Generate personalized email"""
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
            temperature=0.7
        )
        
        email_content = response.choices[0].message.content
        
        # Parse subject and body
        lines = email_content.split('\n')
        subject = lines[0].replace('Subject: ', '') if lines else 'Partnership Opportunity'
        body = '\n'.join(lines[2:]) if len(lines) > 2 else email_content
        
        return {"subject": subject, "body": body}
        
    except Exception as e:
        return {"error": f"Email generation failed: {str(e)}"}

def main():
    st.title("🎯 Sales Automation System")
    st.markdown("**AI-Powered Lead Scoring, Research & Email Generation**")
    
    # Initialize system
    if 'sales_system' not in st.session_state:
        st.session_state.sales_system = SalesAutomationSystem()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    openai_key = st.sidebar.text_input(
        "OpenAI API Key", 
        type="password",
        help="Required for company research and email generation"
    )
    
    min_score = st.sidebar.slider(
        "Minimum Conversion Score", 
        0.0, 1.0, 0.7, 0.1,
        help="Only process prospects above this score"
    )
    
    st.session_state.config = {
        'openai_key': openai_key,
        'min_score': min_score
    }
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Import", 
        "🧠 Model Training", 
        "⚡ Process Leads", 
        "📧 Results",
        "📈 Analytics"
    ])
    
    with tab1:
        st.header("Data Import")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("CRM Data")
            crm_file = st.file_uploader(
                "Upload CRM CSV", 
                type=['csv'],
                key="crm_upload",
                help="Expected columns: company_name, industry, company_size, deal_value, time_to_close, won, location"
            )
            
            if crm_file:
                try:
                    crm_df = pd.read_csv(crm_file)
                    crm_df = st.session_state.sales_system.process_crm_data(crm_df)
                    st.session_state.crm_data = crm_df
                    
                    st.success(f"✅ Loaded {len(crm_df)} CRM records")
                    st.dataframe(crm_df.head())
                    
                    # Data summary
                    won_rate = (crm_df['won'].sum() / len(crm_df)) * 100
                    st.metric("Conversion Rate", f"{won_rate:.1f}%")
                    
                except Exception as e:
                    st.error(f"Error loading CRM data: {e}")
        
        with col2:
            st.subheader("Lead Forensics Data")
            leads_file = st.file_uploader(
                "Upload Leads CSV", 
                type=['csv'],
                key="leads_upload",
                help="Expected columns: company_name, domain, industry, company_size, pages_visited, session_duration, return_visits"
            )
            
            if leads_file:
                try:
                    leads_df = pd.read_csv(leads_file)
                    st.session_state.leads_data = leads_df
                    
                    st.success(f"✅ Loaded {len(leads_df)} lead records")
                    st.dataframe(leads_df.head())
                    
                except Exception as e:
                    st.error(f"Error loading leads data: {e}")
    
    with tab2:
        st.header("Model Training")
        
        if 'crm_data' not in st.session_state:
            st.warning("Please upload CRM data first")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if st.button("🚀 Train Model", type="primary"):
                    with st.spinner("Training model..."):
                        try:
                            metrics = st.session_state.sales_system.train_model(st.session_state.crm_data)
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
                                precision = metrics['classification_report']['weighted avg']['precision']
                                st.metric("Precision", f"{precision:.3f}")
                            
                        except Exception as e:
                            st.error(f"Training failed: {e}")
            
            with col2:
                if st.session_state.model_trained:
                    st.success("✅ Model Ready")
                else:
                    st.info("⏳ Model Not Trained")
    
    with tab3:
        st.header("Process New Leads")
        
        if not st.session_state.model_trained:
            st.warning("Please train the model first")
        elif 'leads_data' not in st.session_state:
            st.warning("Please upload leads data first")
        else:
            batch_size = st.slider("Batch Size", 1, 50, 10)
            
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
                        # Research company
                        research = research_company(
                            lead['company_name'], 
                            lead.get('domain', ''), 
                            openai_key
                        )
                        
                        # Generate email
                        email = generate_email(prospect_data, research, openai_key)
                        
                        results.append({
                            'company_name': lead['company_name'],
                            'industry': lead.get('industry', 'Unknown'),
                            'score': score,
                            'research': research.get('research', 'No research available'),
                            'email_subject': email.get('subject', 'Partnership Opportunity'),
                            'email_body': email.get('body', 'Email generation failed'),
                            'status': 'pending'
                        })
                
                st.session_state.processing_results = results
                progress_bar.progress(1.0)
                status_text.text("✅ Processing complete!")
                
                st.success(f"Processed {len(leads_df)} leads, found {len(results)} qualified prospects")
    
    with tab4:
        st.header("Generated Emails")
        
        if 'processing_results' not in st.session_state:
            st.info("No results yet. Process some leads first.")
        else:
            results = st.session_state.processing_results
            
            for idx, result in enumerate(results):
                with st.expander(f"{result['company_name']} - Score: {result['score']:.2f}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("Generated Email")
                        st.write(f"**Subject:** {result['email_subject']}")
                        st.write(result['email_body'])
                        
                        st.subheader("Company Research")
                        st.write(result['research'])
                    
                    with col2:
                        st.metric("Conversion Score", f"{result['score']:.1%}")
                        st.metric("Industry", result['industry'])
                        
                        if st.button("✅ Approve", key=f"approve_{idx}"):
                            st.session_state.processing_results[idx]['status'] = 'approved'
                            st.success("Email approved!")
                        
                        if st.button("🔄 Regenerate", key=f"regen_{idx}"):
                            with st.spinner("Regenerating..."):
                                new_email = generate_email(
                                    {'company_name': result['company_name'], 'industry': result['industry']},
                                    {'research': result['research']},
                                    openai_key
                                )
                                st.session_state.processing_results[idx]['email_subject'] = new_email.get('subject', 'Partnership Opportunity')
                                st.session_state.processing_results[idx]['email_body'] = new_email.get('body', 'Email generation failed')
                                st.rerun()
            
            # Export functionality
            if st.button("📤 Export Results"):
                df_results = pd.DataFrame(results)
                csv = df_results.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="sales_results.csv">Download CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
    
    with tab5:
        st.header("Analytics Dashboard")
        
        if 'processing_results' in st.session_state:
            results = st.session_state.processing_results
            
            if results:
                # Score distribution
                scores = [r['score'] for r in results]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Prospects", len(results))
                with col2:
                    st.metric("Avg Score", f"{np.mean(scores):.1%}")
                with col3:
                    approved_count = len([r for r in results if r.get('status') == 'approved'])
                    st.metric("Approved Emails", approved_count)
                
                # Score histogram
                st.subheader("Score Distribution")
                fig_data = pd.DataFrame({'Score': scores})
                st.bar_chart(fig_data['Score'])
                
                # Industry breakdown
                industries = [r['industry'] for r in results]
                industry_counts = pd.Series(industries).value_counts()
                
                st.subheader("Industry Breakdown")
                st.bar_chart(industry_counts)
        else:
            st.info("No analytics data available yet.")

if __name__ == "__main__":
    main()