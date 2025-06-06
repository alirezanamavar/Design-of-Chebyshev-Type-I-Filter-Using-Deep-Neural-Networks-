# Save the trained model
torch.save(model.state_dict(), 'chebyshev_model.pth')

# Save scalers
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')
print("Model and scalers saved successfully.")
برای بارگذاری مجدد در آینده:
# Reload model
loaded_model = ChebyshevNet()
loaded_model.load_state_dict(torch.load('chebyshev_model.pth'))
loaded_model.eval()

# Reload scalers
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')
