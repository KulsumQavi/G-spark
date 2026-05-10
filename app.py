from flask import Flask, render_template, request, redirect, url_for, flash
import sqlite3
from datetime import datetime
import random
import re
import joblib
import numpy as np
import pandas as pd
import os
import math
import json


app = Flask(__name__)
app.secret_key = 'farmlinks-secret-2024'

# ========== LOAD ML MODELS ==========
ML_LOADED = False
try:
    if os.path.exists('ml/models/driver_matcher.pkl'):
        driver_model = joblib.load('ml/models/driver_matcher.pkl')
        price_model = joblib.load('ml/models/price_forecaster.pkl')
        spoilage_model = joblib.load('ml/models/spoilage_classifier.pkl')
        crop_list = joblib.load('ml/models/crop_list.pkl')
        ML_LOADED = True
        print("✅ ML Models loaded successfully!")
    else:
        print("⚠️ ML models not found. Run: python ml/train_models.py")
except Exception as e:
    print(f"⚠️ ML load error: {e}")

app = Flask(__name__)

# ========== DATABASE SETUP ==========
def get_db():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        farmer_name TEXT,
        phone TEXT,
        crop_type TEXT,
        quantity INTEGER,
        pickup_location TEXT,
        destination TEXT,
        pickup_time TEXT,
        notes TEXT,
        status TEXT DEFAULT 'Pending',
        driver_name TEXT,
        driver_vehicle TEXT,
        driver_rating REAL,
        driver_phone TEXT,
        estimated_cost INTEGER,
        created_at TEXT
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS drivers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        phone TEXT,
        vehicle TEXT,
        capacity INTEGER,
        rating REAL,
        points INTEGER,
        location TEXT,
        tier TEXT,
        available INTEGER DEFAULT 1
    )''')

    c.execute("SELECT COUNT(*) FROM drivers")
    if c.fetchone()[0] == 0:
        sample_drivers = [
            ('Suresh Kumar',  '+91 98765 43210', 'MH-12-AB-1234', 1000, 4.8, 2450, 'Nashik', 'Silver', 1),
            ('Ramesh Patil',  '+91 98765 43211', 'MH-14-CD-5678', 1500, 4.5, 1800, 'Pune',   'Silver', 1),
            ('Vijay Singh',   '+91 98765 43212', 'MH-15-EF-9012',  800, 4.9, 3200, 'Nashik', 'Gold',   1),
            ('Anil Sharma',   '+91 98765 43213', 'MH-12-GH-3456', 2000, 4.2,  950, 'Mumbai', 'Bronze', 1),
            ('Mukesh Yadav',  '+91 98765 43214', 'MH-13-IJ-7890', 1200, 4.7, 2100, 'Pune',   'Silver', 1),
        ]
        c.executemany(
            "INSERT INTO drivers (name,phone,vehicle,capacity,rating,points,location,tier,available) VALUES (?,?,?,?,?,?,?,?,?)",
            sample_drivers
        )

    conn.commit()
    conn.close()


# ========== HELPERS ==========
def find_best_driver(quantity, location):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT * FROM drivers WHERE available=1 AND capacity >= ?", (quantity,))
    drivers = c.fetchall()
    conn.close()
    if not drivers:
        return None
    best, best_score = None, 0
    for d in drivers:
        score = (d[5] / 5) * 30 + min(d[6] / 100, 30)
        score += 40 if d[7].lower() in location.lower() or location.lower() in d[7].lower() else 20
        if score > best_score:
            best_score, best = score, d
    return best

def calculate_cost(quantity, distance_km=45):
    return int(quantity * 8 * 0.4)

def update_driver_tier(conn, driver_id):
    c = conn.cursor()
    c.execute("SELECT points FROM drivers WHERE id=?", (driver_id,))
    row = c.fetchone()
    if not row:
        return
    pts = row[0]
    if pts >= 6000:   tier = 'Diamond'
    elif pts >= 3000: tier = 'Gold'
    elif pts >= 1000: tier = 'Silver'
    else:             tier = 'Bronze'
    c.execute("UPDATE drivers SET tier=? WHERE id=?", (tier, driver_id))


# ========== ML INFERENCE ==========
def ml_predict_driver_score(rating, points, capacity, distance, urgency, past_completed):
    if not ML_LOADED:
        return rating * 20
    features = pd.DataFrame([{'rating': rating, 'points': points, 'capacity': capacity,
                               'distance': distance, 'urgency': urgency, 'past_completed': past_completed}])
    return round(float(driver_model.predict(features)[0]), 2)

def ml_predict_price(crop_name, days_ahead=7):
    if not ML_LOADED or crop_name not in crop_list:
        return [30] * days_ahead
    crop_id = crop_list.index(crop_name)
    today = datetime.now().timetuple().tm_yday
    predictions = []
    for day in range(1, days_ahead + 1):
        future_day = (today + day) % 365
        features = pd.DataFrame([{'crop_id': crop_id, 'day': future_day,
                                   'demand': random.uniform(0.7, 1.3),
                                   'supply': random.uniform(0.7, 1.3),
                                   'weather': random.uniform(0.8, 1.1),
                                   'season': future_day / 365}])
        predictions.append(round(float(price_model.predict(features)[0]), 2))
    return predictions

def ml_predict_spoilage(temperature, humidity, hours, perishability, distance):
    if not ML_LOADED:
        return {'risk': 0, 'label': 'Safe', 'confidence': 0.5}
    features = pd.DataFrame([{'temperature': temperature, 'humidity': humidity,
                               'hours_since_pickup': hours, 'perishability': perishability,
                               'distance_remaining': distance}])
    risk = int(spoilage_model.predict(features)[0])
    proba = spoilage_model.predict_proba(features)[0]
    labels = {0: '✅ Safe', 1: '⚠️ Warning', 2: '🚨 Critical'}
    return {'risk': risk, 'label': labels[risk], 'confidence': round(float(max(proba)) * 100, 1),
            'probabilities': {'safe': round(float(proba[0]) * 100, 1),
                              'warning': round(float(proba[1]) * 100, 1),
                              'critical': round(float(proba[2]) * 100, 1)}}


# ========== ROUTES ==========

@app.route('/')
def home():
    return render_template('home.html')


# ---------- FARMER ----------
@app.route('/farmer', methods=['GET', 'POST'])
def farmer():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    if request.method == 'POST':
        try:
            farmer_name  = request.form['farmer_name']
            phone        = request.form['phone']
            crop         = request.form['crop_type']
            qty          = int(request.form['quantity'])
            location     = request.form['pickup_location']
            destination  = request.form['destination']
            pickup_time  = request.form['pickup_time']
            notes        = request.form.get('notes', '')
            cost         = calculate_cost(qty)

            # Save as Pending — driver will accept from their dashboard
            c.execute('''
                INSERT INTO requests
                (farmer_name, phone, crop_type, quantity, pickup_location, destination,
                 pickup_time, notes, status, driver_name, driver_vehicle, driver_rating,
                 driver_phone, estimated_cost, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (farmer_name, phone, crop, qty, location, destination,
                  pickup_time, notes, 'Pending',
                  None, None, None, None, cost,
                  datetime.now().strftime('%d %b %Y, %I:%M %p')))

            request_id = c.lastrowid
            conn.commit()

            c.execute("SELECT * FROM requests WHERE id=?", (request_id,))
            req = c.fetchone()
            conn.close()
            return render_template('farmer_success.html', req=req)

        except Exception as e:
            conn.close()
            return f"Farmer form error: {e}"

    c.execute("SELECT * FROM requests ORDER BY id DESC LIMIT 5")
    requests_list = c.fetchall()
    conn.close()
    return render_template('farmer.html', requests=requests_list)


# ---------- DRIVER ----------
@app.route('/driver')
def driver():
    driver_id = request.args.get('id', 1, type=int)
    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    c.execute("SELECT * FROM drivers WHERE id=?", (driver_id,))
    driver_info = c.fetchone()

    c.execute("SELECT * FROM drivers ORDER BY points DESC")
    all_drivers = c.fetchall()

    # Only truly unassigned requests visible to all drivers
    c.execute("SELECT * FROM requests WHERE status='Pending' ORDER BY id DESC")
    available = c.fetchall()

    # Active = accepted/in-progress BY THIS driver
    c.execute("""SELECT * FROM requests
                 WHERE status IN ('Driver Assigned','In Progress')
                 AND driver_name = (SELECT name FROM drivers WHERE id=?)
                 ORDER BY id DESC""", (driver_id,))
    active = c.fetchall()

    # Completed BY THIS driver
    c.execute("""SELECT * FROM requests
                 WHERE status='Completed'
                 AND driver_name = (SELECT name FROM drivers WHERE id=?)
                 ORDER BY id DESC LIMIT 10""", (driver_id,))
    completed = c.fetchall()

    points = driver_info[6]
    if points >= 6000:
        tier_name, tier_color, next_tier_points = "💎 Diamond", "#00BCD4", 10000
    elif points >= 3000:
        tier_name, tier_color, next_tier_points = "🥇 Gold",    "#FFD700",  6000
    elif points >= 1000:
        tier_name, tier_color, next_tier_points = "🥈 Silver",  "#9E9E9E",  3000
    else:
        tier_name, tier_color, next_tier_points = "🥉 Bronze",  "#CD7F32",  1000

    progress_percent = min((points / next_tier_points) * 100, 100)
    total_earnings   = len(completed) * 1200
    leaderboard      = all_drivers[:5]

    conn.close()
    return render_template('driver.html',
        driver=driver_info, all_drivers=all_drivers,
        available=available, active=active, completed=completed,
        tier_name=tier_name, tier_color=tier_color,
        next_tier_points=next_tier_points, progress_percent=progress_percent,
        total_earnings=total_earnings, leaderboard=leaderboard)


@app.route('/driver/accept/<int:job_id>')
def accept_job(job_id):
    driver_id = request.args.get('driver_id', 1, type=int)
    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    # Guard: only accept if still Pending
    c.execute("SELECT status FROM requests WHERE id=?", (job_id,))
    row = c.fetchone()
    if row and row[0] == 'Pending':
        c.execute("SELECT * FROM drivers WHERE id=?", (driver_id,))
        drv = c.fetchone()
        c.execute("""UPDATE requests
                     SET status='Driver Assigned',
                         driver_name=?, driver_vehicle=?, driver_rating=?, driver_phone=?
                     WHERE id=?""",
                  (drv[1], drv[3], drv[5], drv[2], job_id))
        c.execute("UPDATE drivers SET points = points + 50 WHERE id=?", (driver_id,))
        update_driver_tier(conn, driver_id)
        conn.commit()

    conn.close()
    return redirect(url_for('driver', id=driver_id))


@app.route('/driver/decline/<int:job_id>')
def decline_job(job_id):
    driver_id = request.args.get('driver_id', 1, type=int)
    # No DB change — just send driver back; another driver can still pick it up
    return redirect(url_for('driver', id=driver_id))


@app.route('/driver/start/<int:job_id>')
def start_job(job_id):
    driver_id = request.args.get('driver_id', 1, type=int)
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("""UPDATE requests SET status='In Progress'
                 WHERE id=? AND driver_name=(SELECT name FROM drivers WHERE id=?)""",
              (job_id, driver_id))
    conn.commit()
    conn.close()
    return redirect(url_for('driver', id=driver_id))


@app.route('/driver/complete/<int:job_id>')
def complete_job(job_id):
    driver_id = request.args.get('driver_id', 1, type=int)
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("""UPDATE requests SET status='Completed'
                 WHERE id=? AND driver_name=(SELECT name FROM drivers WHERE id=?)""",
              (job_id, driver_id))
    c.execute("UPDATE drivers SET points = points + 100 WHERE id=?", (driver_id,))
    update_driver_tier(conn, driver_id)
    conn.commit()
    conn.close()
    return redirect(url_for('driver', id=driver_id))


# ---------- BUYER ----------
def predict_price(crop, current_price):
    volatility = {'Tomatoes': 0.15, 'Chillies': 0.18, 'Mangoes': 0.20, 'Bananas': 0.12,
                  'Onions': 0.10, 'Potatoes': 0.08, 'Wheat': 0.04, 'Rice': 0.03}
    vol   = volatility.get(crop, 0.08)
    trend = 0.02 if crop in ['Tomatoes', 'Chillies', 'Mangoes'] else 0.005
    base  = current_price
    preds = []
    for day in range(1, 8):
        wave  = math.sin(day * 0.7) * vol * base
        noise = random.uniform(-vol / 2, vol / 2) * base
        preds.append(round(max(base + wave + trend * day * base + noise, base * 0.7), 2))
    return preds

def get_market_insights(crop):
    insights = {
        'Tomatoes': {'demand': 'High',      'trend': '📈 Rising', 'tip': 'Buy now! Prices expected to rise 15% next week'},
        'Onions':   {'demand': 'Medium',    'trend': '➡️ Stable', 'tip': 'Stable supply. Good time to stock up.'},
        'Potatoes': {'demand': 'High',      'trend': '📉 Falling','tip': 'Wait 3-4 days. Prices dropping due to harvest.'},
        'Chillies': {'demand': 'Very High', 'trend': '📈 Rising', 'tip': 'URGENT: Buy today! Major shortage coming.'},
        'Wheat':    {'demand': 'Medium',    'trend': '➡️ Stable', 'tip': 'Steady prices. Bulk orders recommended.'},
        'Rice':     {'demand': 'High',      'trend': '➡️ Stable', 'tip': 'Reliable supply. Long-term contracts ideal.'},
        'Mangoes':  {'demand': 'Very High', 'trend': '📈 Rising', 'tip': 'Season ending! Stock now for premium prices.'},
        'Bananas':  {'demand': 'Medium',    'trend': '📉 Falling','tip': 'Oversupply. Wait for better deals.'},
    }
    return insights.get(crop, {'demand': 'Medium', 'trend': '➡️ Stable', 'tip': 'Normal market conditions.'})

@app.route('/buyer')
def buyer():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    # Include Pending so newly submitted farmer requests are visible
    c.execute("""SELECT * FROM requests
                 WHERE status IN ('Pending','Driver Assigned','In Progress','Completed')
                 ORDER BY id DESC""")
    available_crops = c.fetchall()
    conn.close()

    market_prices = {
        'Tomatoes': 35, 'Onions': 28, 'Potatoes': 22, 'Chillies': 80,
        'Wheat': 32, 'Rice': 45, 'Mangoes': 120, 'Bananas': 40
    }
    forecasts = {
        crop: {
            'current': price,
            'predictions': predict_price(crop, price),
            'insights': get_market_insights(crop)
        }
        for crop, price in market_prices.items()
    }
    return render_template('buyer.html',
                           available_crops=available_crops,
                           forecasts=forecasts,
                           high_demand_crops=['Chillies', 'Tomatoes', 'Mangoes'],
                           market_prices=market_prices)
@app.route('/buyer/order/<int:req_id>')
def buyer_order(req_id):
    qty = request.args.get('qty', 25, type=int)

    # Enforce minimum server-side
    if qty < 25:
        flash('⚠️ Minimum order quantity is 25 kg.')
        return redirect(url_for('buyer'))

    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    c.execute("SELECT status, crop_type, quantity FROM requests WHERE id=?", (req_id,))
    row = c.fetchone()

    if not row:
        flash('⚠️ This listing no longer exists.')
        conn.close()
        return redirect(url_for('buyer'))

    if row[0] == 'Sold':
        flash('⚠️ This lot has already been sold.')
        conn.close()
        return redirect(url_for('buyer'))

    available_qty = row[2]

    if qty > available_qty:
        flash(f'⚠️ Only {available_qty} kg available for this lot.')
        conn.close()
        return redirect(url_for('buyer'))

    if qty >= available_qty:
        # Buyer takes the full lot
        c.execute("UPDATE requests SET status='Sold' WHERE id=?", (req_id,))
        flash(f"✅ Full lot ordered! {row[1]} ({available_qty} kg) purchased successfully. Delivery within 24 hrs.")
    else:
        # Partial buy — reduce quantity, keep listing active
        new_qty = available_qty - qty
        c.execute("UPDATE requests SET quantity=? WHERE id=?", (new_qty, req_id))
        flash(f"✅ Order placed! {row[1]} ({qty} kg) purchased. {new_qty} kg still available from this lot.")

    conn.commit()
    conn.close()
    return redirect(url_for('buyer'))


# ---------- IoT ----------
def generate_iot_data(crop_type, status):
    optimal = {
        'Tomatoes': {'temp': (8,12),  'humidity': (85,95)},
        'Onions':   {'temp': (0,5),   'humidity': (65,75)},
        'Potatoes': {'temp': (4,8),   'humidity': (90,95)},
        'Chillies': {'temp': (7,10),  'humidity': (90,95)},
        'Wheat':    {'temp': (15,25), 'humidity': (40,60)},
        'Rice':     {'temp': (15,25), 'humidity': (45,65)},
        'Mangoes':  {'temp': (10,13), 'humidity': (85,90)},
        'Bananas':  {'temp': (13,15), 'humidity': (85,95)},
    }
    opt  = optimal.get(crop_type, {'temp': (10,20), 'humidity': (70,85)})
    temp = round(random.uniform(opt['temp'][0] - 2, opt['temp'][1] + 3), 1)
    hum  = round(random.uniform(opt['humidity'][0] - 5, opt['humidity'][1] + 2), 1)
    lat  = round(19.9975 + random.uniform(-0.5, 0.5), 4)
    lng  = round(73.7898 + random.uniform(-0.5, 0.5), 4)
    speed   = round(random.uniform(40, 70), 1) if status == 'In Progress' else 0
    battery = random.randint(60, 100)

    t_ok  = opt['temp'][0] <= temp <= opt['temp'][1]
    h_ok  = opt['humidity'][0] <= hum <= opt['humidity'][1]
    t_status = 'optimal' if t_ok else ('warning' if abs(temp - sum(opt['temp'])/2) < 5 else 'critical')
    h_status = 'optimal' if h_ok else ('warning' if abs(hum - sum(opt['humidity'])/2) < 10 else 'critical')

    alerts = []
    if t_status == 'critical': alerts.append({'type':'danger',  'msg': f'⚠️ Temperature {temp}°C outside safe range!'})
    elif t_status == 'warning': alerts.append({'type':'warning', 'msg': f'⚡ Temperature {temp}°C approaching limit'})
    if h_status == 'critical': alerts.append({'type':'danger',  'msg': f'💧 Humidity {hum}% critical!'})
    if battery < 20:           alerts.append({'type':'warning', 'msg': f'🔋 Sensor battery low ({battery}%)'})
    if not alerts:             alerts.append({'type':'success', 'msg': '✅ All sensors normal'})

    return {'temperature': temp, 'humidity': hum, 'lat': lat, 'lng': lng,
            'speed': speed, 'battery': battery,
            'temp_status': t_status, 'humidity_status': h_status,
            'optimal_temp': f"{opt['temp'][0]}-{opt['temp'][1]}°C",
            'optimal_humidity': f"{opt['humidity'][0]}-{opt['humidity'][1]}%",
            'alerts': alerts, 'timestamp': datetime.now().strftime('%H:%M:%S')}

@app.route('/iot')
def iot_dashboard():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT * FROM requests WHERE status IN ('Driver Assigned','In Progress') ORDER BY id DESC")
    active_deliveries = c.fetchall()
    conn.close()
    iot_data = {d[0]: generate_iot_data(d[3], d[9]) for d in active_deliveries}
    return render_template('iot.html', deliveries=active_deliveries, iot_data=iot_data)

@app.route('/api/iot/<int:req_id>')
def api_iot(req_id):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT * FROM requests WHERE id=?", (req_id,))
    delivery = c.fetchone()
    conn.close()
    if delivery:
        return json.dumps(generate_iot_data(delivery[3], delivery[9]))
    return json.dumps({'error': 'Not found'})


# ---------- AI BRAIN ----------
def parse_farmer_message(message):
    msg = message.lower()
    crops = {'tomato':'Tomatoes','tamatar':'Tomatoes','onion':'Onions','pyaaz':'Onions',
             'kanda':'Onions','potato':'Potatoes','aloo':'Potatoes','batata':'Potatoes',
             'chilli':'Chillies','mirchi':'Chillies','mirch':'Chillies',
             'wheat':'Wheat','gehu':'Wheat','gehun':'Wheat',
             'rice':'Rice','chawal':'Rice','dhan':'Rice',
             'mango':'Mangoes','aam':'Mangoes','banana':'Bananas','kela':'Bananas'}
    detected_crop = next((v for k, v in crops.items() if k in msg), 'Unknown')
    qty_match = re.search(r'(\d+)\s*(kg|kilo|quintal|ton|tonne)?', msg)
    quantity = int(qty_match.group(1)) if qty_match else 100
    if qty_match and qty_match.group(2) == 'quintal': quantity *= 100
    elif qty_match and qty_match.group(2) in ('ton','tonne'): quantity *= 1000
    locations = ['nashik','pune','mumbai','rampur','igatpuri','sinnar','malegaon','satara','kolhapur']
    detected_location = next((l.title() for l in locations if l in msg), 'Unknown Village')
    urgency = 'High' if any(w in msg for w in ['urgent','asap','now','jaldi','turant','immediate']) \
              or detected_crop in ['Tomatoes','Chillies','Mangoes','Bananas'] else 'Normal'
    spoilage_windows = {'Tomatoes':'4 hours','Chillies':'6 hours','Mangoes':'12 hours',
                        'Bananas':'12 hours','Onions':'48 hours','Potatoes':'72 hours',
                        'Wheat':'7 days','Rice':'7 days'}
    return {'crop': detected_crop, 'quantity': quantity, 'location': detected_location,
            'urgency': urgency, 'spoilage_window': spoilage_windows.get(detected_crop,'24 hours'),
            'priority_score': min(10, {'Tomatoes':9,'Chillies':9,'Mangoes':8,'Bananas':8,
                                       'Onions':5,'Potatoes':4,'Wheat':2,'Rice':2}.get(detected_crop,5)
                                  + (1 if urgency == 'High' else 0))}

def find_clustered_farmers(location, crop):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT * FROM requests WHERE pickup_location LIKE ? OR crop_type=? ORDER BY id DESC LIMIT 5",
              (f'%{location}%', crop))
    nearby = c.fetchall()
    conn.close()
    return nearby

@app.route('/ai-brain', methods=['GET', 'POST'])
def ai_brain():
    parsed_data, nearby_farmers = None, []
    sample_message = "500 kg tomatoes urgent from Rampur Nashik"
    if request.method == 'POST':
        message = request.form.get('message', '')
        sample_message = message
        parsed_data = parse_farmer_message(message)
        nearby_farmers = find_clustered_farmers(parsed_data['location'], parsed_data['crop'])
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT pickup_location, COUNT(*) FROM requests GROUP BY pickup_location ORDER BY COUNT(*) DESC LIMIT 5")
    clusters = c.fetchall()
    c.execute("SELECT crop_type, COUNT(*) FROM requests GROUP BY crop_type ORDER BY COUNT(*) DESC")
    crop_distribution = c.fetchall()
    conn.close()
    return render_template('ai_brain.html', parsed=parsed_data, nearby=nearby_farmers,
                           sample_message=sample_message, clusters=clusters,
                           crop_distribution=crop_distribution)


# ---------- ML LAB ----------
@app.route('/ml-lab', methods=['GET', 'POST'])
def ml_lab():
    result, test_type = None, request.args.get('test', 'driver')
    if request.method == 'POST':
        test_type = request.form.get('test_type')
        if test_type == 'driver':
            score = ml_predict_driver_score(
                float(request.form['rating']), int(request.form['points']),
                int(request.form['capacity']), float(request.form['distance']),
                int(request.form['urgency']), int(request.form['past_completed']))
            result = {'type': 'driver', 'score': score}
        elif test_type == 'price':
            crop = request.form['crop']
            result = {'type': 'price', 'crop': crop, 'predictions': ml_predict_price(crop, 7)}
        elif test_type == 'spoilage':
            result = {'type': 'spoilage', 'data': ml_predict_spoilage(
                float(request.form['temperature']), float(request.form['humidity']),
                float(request.form['hours']), int(request.form['perishability']),
                float(request.form['distance']))}
    return render_template('ml_lab.html', result=result, test_type=test_type, ml_loaded=ML_LOADED)


# ---------- MAPS ----------
def haversine_distance(lat1, lng1, lat2, lng2):
    R = 6371
    lat1, lat2 = math.radians(lat1), math.radians(lat2)
    dlat, dlng = lat2 - lat1, math.radians(lng2 - lng1)
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlng/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def optimize_route(start, stops, end):
    if not stops:
        return [start, end]
    route, remaining, current = [start], stops.copy(), start
    while remaining:
        nearest = min(remaining, key=lambda s: haversine_distance(
            current['lat'], current['lng'], s['lat'], s['lng']))
        route.append(nearest); remaining.remove(nearest); current = nearest
    return route + [end]

def calculate_route_stats(route):
    dist = sum(haversine_distance(route[i]['lat'], route[i]['lng'],
                                   route[i+1]['lat'], route[i+1]['lng'])
               for i in range(len(route)-1))
    t = dist / 45
    return {'total_distance': round(dist, 2), 'time_hours': round(t, 2),
            'time_display': f"{int(t)}h {int((t%1)*60)}min",
            'fuel_liters': round(dist*0.12, 2),
            'fuel_cost': round(dist*0.12*95, 2),
            'co2_kg': round(dist*0.27, 2)}

@app.route('/maps')
def maps_dashboard():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT * FROM requests WHERE status IN ('Driver Assigned','In Progress','Pending') ORDER BY id DESC LIMIT 20")
    deliveries = c.fetchall()
    c.execute("SELECT * FROM drivers")
    drivers = c.fetchall()
    conn.close()

    location_coords = {
        'nashik':   {'lat':19.9975,'lng':73.7898,'name':'Nashik'},
        'pune':     {'lat':18.5204,'lng':73.8567,'name':'Pune'},
        'mumbai':   {'lat':19.0760,'lng':72.8777,'name':'Mumbai'},
        'igatpuri': {'lat':19.6967,'lng':73.5634,'name':'Igatpuri'},
        'sinnar':   {'lat':19.8467,'lng':74.0000,'name':'Sinnar'},
        'malegaon': {'lat':20.5500,'lng':74.5333,'name':'Malegaon'},
        'satara':   {'lat':17.6805,'lng':74.0183,'name':'Satara'},
        'kolhapur': {'lat':16.7050,'lng':74.2433,'name':'Kolhapur'},
        'rampur':   {'lat':19.7500,'lng':73.8500,'name':'Rampur'},
    }

    farmer_markers = []
    for d in deliveries:
        loc_key = d[5].lower().split(',')[0].strip()
        coords = next((v for k, v in location_coords.items() if k in loc_key or loc_key in k),
                      {'lat': 19.9975 + random.uniform(-0.3,0.3),
                       'lng': 73.7898 + random.uniform(-0.3,0.3), 'name': d[5]})
        farmer_markers.append({'id':d[0], 'lat':coords['lat']+random.uniform(-0.05,0.05),
                                'lng':coords['lng']+random.uniform(-0.05,0.05),
                                'name':d[1],'crop':d[3],'quantity':d[4],
                                'location':d[5],'destination':d[6],
                                'status':d[9],'driver':d[10] or 'Unassigned'})

    driver_markers = []
    for d in drivers:
        coords = location_coords.get(d[7].lower(), {'lat':19.9975,'lng':73.7898})
        driver_markers.append({'id':d[0],'lat':coords['lat']+random.uniform(-0.1,0.1),
                                'lng':coords['lng']+random.uniform(-0.1,0.1),
                                'name':d[1],'vehicle':d[3],'capacity':d[4],
                                'rating':d[5],'tier':d[8],'location':d[7]})

    route_demo = None
    if len(farmer_markers) >= 3:
        start = {'lat':19.9975,'lng':73.7898,'name':'Driver Start (Nashik)'}
        end   = {'lat':18.5204,'lng':73.8567,'name':'Pune APMC Market'}
        opt   = optimize_route(start, farmer_markers[:3], end)
        unopt = [start] + farmer_markers[:3] + [end]
        stats, unopt_stats = calculate_route_stats(opt), calculate_route_stats(unopt)
        saved = unopt_stats['total_distance'] - stats['total_distance']
        route_demo = {'optimized':opt, 'stats':stats, 'unopt_stats':unopt_stats,
                      'savings_km': round(saved, 2),
                      'savings_pct': round(saved/unopt_stats['total_distance']*100, 1) if unopt_stats['total_distance'] else 0}

    return render_template('maps.html', farmers=farmer_markers, drivers=driver_markers, route_demo=route_demo)


# ========== RUN ==========
if __name__ == '__main__':
    init_db()
    app.run(debug=True)