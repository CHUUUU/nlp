from service import app

if __name__ == "__main__":
    host = "0.0.0.0"
    port = "9500" 
    
    app.run(host=host, port=port)
