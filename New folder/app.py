# app.py
import os
import ssl
import urllib.request
import uuid
import hashlib
import datetime
import json
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, abort
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from sqlalchemy import func, extract

try:
    from PIL import Image
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    AI_IMPORT_ERROR = None
except Exception as exc:
    Image = None
    torch = None
    models = None
    transforms = None
    AI_IMPORT_ERROR = exc

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'waste_management_secret_key_2025')

database_url = os.environ.get('DATABASE_URL')
if database_url and database_url.startswith('postgres://'):
    database_url = database_url.replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_DATABASE_URI'] = database_url or 'sqlite:///waste_management.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', os.path.join(app.root_path, 'static', 'uploads'))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

db = SQLAlchemy(app)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ------------------------- AI Waste Classification Model -------------------------
# Mapping ImageNet labels to waste categories
waste_mapping = {
    'plastic': ['plastic_bag', 'plastic_bottle', 'bucket', 'polymer', 'jar', 'container', 'packet'],
    'paper': ['book', 'paper', 'cardboard', 'envelope', 'newspaper', 'magazine', 'tissue', 'box'],
    'glass': ['glass', 'wine_bottle', 'bottle', 'jar', 'mirror', 'window'],
    'metal': ['can', 'tin', 'metal', 'cooking_pan', 'aluminum', 'bottle_cap'],
    'organic': ['banana', 'apple', 'fruit', 'vegetable', 'leaf', 'food', 'bread', 'cake', 'egg'],
    'ewaste': ['computer', 'laptop', 'phone', 'keyboard', 'mouse', 'monitor', 'cable', 'circuit']
}

# Cache ImageNet labels to avoid repeated downloads
_imagenet_labels = None
device = None
model = None
preprocess = None

def get_ai_model():
    """Load the AI model only when image classification is actually used."""
    global device, model, preprocess
    if AI_IMPORT_ERROR is not None:
        print(f"AI libraries not available: {AI_IMPORT_ERROR}")
        return None

    if model is None:
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            model.eval()
            model = model.to(device)
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        except Exception as exc:
            print(f"AI model load error: {exc}")
            return None

    return model

def get_imagenet_labels():
    global _imagenet_labels
    if _imagenet_labels is None:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        try:
            with urllib.request.urlopen("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", context=ctx) as f:
                _imagenet_labels = [line.decode('utf-8').strip() for line in f.readlines()]
        except:
            # Fallback labels if download fails
            _imagenet_labels = ["object"] * 1000
    return _imagenet_labels

def classify_waste_type(image_path):
    """AI-based waste classification using ResNet18"""
    try:
        ai_model = get_ai_model()
        if ai_model is None or Image is None:
            return "Mixed Waste"

        image = Image.open(image_path).convert('RGB')
        input_tensor = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = ai_model(input_tensor)
        
        _, predicted_idx = torch.max(output, 1)
        predicted_idx = predicted_idx.item()
        
        labels = get_imagenet_labels()
        predicted_class = labels[predicted_idx] if predicted_idx < len(labels) else "object"
        predicted_class_lower = predicted_class.lower()
        
        # Map to waste type
        for waste_type, keywords in waste_mapping.items():
            if any(keyword in predicted_class_lower for keyword in keywords):
                return waste_type.capitalize()
        
        # Default fallback
        return "Recyclable" if ("box" in predicted_class_lower or "container" in predicted_class_lower) else "General Waste"
    
    except Exception as e:
        print(f"AI classification error: {e}")
        return "Mixed Waste"

# ------------------------- Database Models -------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    phone = db.Column(db.String(15), nullable=False)
    role = db.Column(db.String(20), default='user')  # user, collector, admin
    points = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    
    addresses = db.relationship('Address', backref='user', lazy=True, cascade='all, delete-orphan')
    pickups = db.relationship('PickupRequest', backref='user', lazy=True, foreign_keys='PickupRequest.user_id')
    collector_pickups = db.relationship('PickupRequest', backref='collector', lazy=True, foreign_keys='PickupRequest.collector_id')
    complaints = db.relationship('Complaint', backref='user', lazy=True)
    notifications = db.relationship('Notification', backref='user', lazy=True)

class Address(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    address_line = db.Column(db.String(255), nullable=False)
    city = db.Column(db.String(100), nullable=False)
    state = db.Column(db.String(100), nullable=False)
    pincode = db.Column(db.String(10), nullable=False)
    area = db.Column(db.String(100), nullable=False)
    landmark = db.Column(db.String(255))
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    is_default = db.Column(db.Boolean, default=False)

class PickupRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    collector_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    address_id = db.Column(db.Integer, db.ForeignKey('address.id'), nullable=False)
    waste_type = db.Column(db.String(50), nullable=False)
    waste_image = db.Column(db.String(255))
    scheduled_date = db.Column(db.Date, nullable=False)
    scheduled_time = db.Column(db.String(20), nullable=False)
    status = db.Column(db.String(20), default='pending')  # pending, accepted, in_progress, completed, cancelled
    points_earned = db.Column(db.Integer, default=0)
    collector_lat = db.Column(db.Float, nullable=True)
    collector_lng = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    completed_at = db.Column(db.DateTime, nullable=True)
    weight_estimate = db.Column(db.Float, default=1.0)
    
    address = db.relationship('Address', backref='pickups')

class RewardTransaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    points = db.Column(db.Integer, nullable=False)  # positive for earning, negative for redemption
    action = db.Column(db.String(50), nullable=False)  # earned, redeemed
    description = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message = db.Column(db.String(500), nullable=False)
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class Complaint(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    subject = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), default='pending')  # pending, resolved
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# ------------------------- Helper Functions -------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login first', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def role_required(*roles):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                flash('Please login first', 'danger')
                return redirect(url_for('login'))
            user = User.query.get(session['user_id'])
            if user.role not in roles:
                flash('Access denied', 'danger')
                return redirect(url_for('dashboard'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def create_notification(user_id, message):
    notif = Notification(user_id=user_id, message=message)
    db.session.add(notif)
    db.session.commit()

# ------------------------- Routes -------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = hashlib.sha256(request.form['password'].encode()).hexdigest()
        phone = request.form['phone']
        role = 'user'
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'danger')
            return redirect(url_for('register'))
        
        user = User(name=name, email=email, password=password, phone=phone, role=role)
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = hashlib.sha256(request.form['password'].encode()).hexdigest()
        user = User.query.filter_by(email=email, password=password).first()
        
        if user:
            session['user_id'] = user.id
            session['role'] = user.role
            flash(f'Welcome back, {user.name}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    user = User.query.get(session['user_id'])
    if user.role == 'admin':
        return redirect(url_for('admin_dashboard'))
    elif user.role == 'collector':
        return redirect(url_for('collector_panel'))
    else:
        return redirect(url_for('user_dashboard'))

# ------------------------- User Panel -------------------------
@app.route('/user/dashboard')
@login_required
@role_required('user')
def user_dashboard():
    user = User.query.get(session['user_id'])
    recent_pickups = PickupRequest.query.filter_by(user_id=user.id).order_by(PickupRequest.created_at.desc()).limit(5).all()
    addresses = Address.query.filter_by(user_id=user.id).all()
    notifications = Notification.query.filter_by(user_id=user.id, is_read=False).count()
    return render_template('user_dashboard.html', user=user, recent_pickups=recent_pickups, addresses=addresses, notifications=notifications)

@app.route('/user/addresses', methods=['GET', 'POST'])
@login_required
@role_required('user')
def manage_addresses():
    user = User.query.get(session['user_id'])
    if request.method == 'POST':
        address_line = request.form['address_line']
        city = request.form['city']
        state = request.form['state']
        pincode = request.form['pincode']
        area = request.form['area']
        landmark = request.form.get('landmark', '')
        
        address = Address(
            user_id=user.id,
            address_line=address_line,
            city=city,
            state=state,
            pincode=pincode,
            area=area,
            landmark=landmark,
            is_default=len(Address.query.filter_by(user_id=user.id).all()) == 0
        )
        db.session.add(address)
        db.session.commit()
        flash('Address added successfully', 'success')
        return redirect(url_for('manage_addresses'))
    
    addresses = Address.query.filter_by(user_id=user.id).all()
    return render_template('address.html', addresses=addresses)

@app.route('/user/address/delete/<int:address_id>')
@login_required
@role_required('user')
def delete_address(address_id):
    address = Address.query.get_or_404(address_id)
    if address.user_id != session['user_id']:
        abort(403)
    db.session.delete(address)
    db.session.commit()
    flash('Address deleted', 'success')
    return redirect(url_for('manage_addresses'))

@app.route('/user/book_pickup', methods=['GET', 'POST'])
@login_required
@role_required('user')
def book_pickup():
    user = User.query.get(session['user_id'])
    addresses = Address.query.filter_by(user_id=user.id).all()
    
    if not addresses:
        flash('Please add an address first', 'warning')
        return redirect(url_for('manage_addresses'))
    
    if request.method == 'POST':
        address_id = request.form['address_id']
        waste_type = request.form['waste_type']
        scheduled_date = datetime.datetime.strptime(request.form['scheduled_date'], '%Y-%m-%d').date()
        scheduled_time = request.form['scheduled_time']
        
        waste_image = None
        if 'waste_image' in request.files:
            file = request.files['waste_image']
            if file and allowed_file(file.filename):
                filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                waste_image = filename
                
                # AI Waste Classification
                predicted_waste = classify_waste_type(filepath)
                if waste_type == 'other' or waste_type == '':
                    waste_type = predicted_waste
                    flash(f'AI identified waste type as: {predicted_waste}', 'info')
        
        pickup = PickupRequest(
            user_id=user.id,
            address_id=address_id,
            waste_type=waste_type,
            waste_image=waste_image,
            scheduled_date=scheduled_date,
            scheduled_time=scheduled_time,
            status='pending'
        )
        db.session.add(pickup)
        db.session.commit()
        
        create_notification(user.id, f'Pickup request #{pickup.id} created successfully. Waiting for collector assignment.')
        flash('Pickup booked successfully!', 'success')
        return redirect(url_for('user_dashboard'))
    
    return render_template('book_pickup.html', addresses=addresses)

@app.route('/user/pickups')
@login_required
@role_required('user')
def user_pickups():
    user = User.query.get(session['user_id'])
    pickups = PickupRequest.query.filter_by(user_id=user.id).order_by(PickupRequest.created_at.desc()).all()
    return render_template('user_pickkups.html', pickups=pickups)

@app.route('/user/tracking/<int:pickup_id>')
@login_required
@role_required('user')
def tracking(pickup_id):
    pickup = PickupRequest.query.get_or_404(pickup_id)
    if pickup.user_id != session['user_id']:
        abort(403)
    return render_template('tracking.html', pickup=pickup)

@app.route('/api/pickup_location/<int:pickup_id>')
@login_required
def get_pickup_location(pickup_id):
    pickup = PickupRequest.query.get_or_404(pickup_id)
    address = pickup.address
    return jsonify({
        'status': pickup.status,
        'collector_lat': pickup.collector_lat,
        'collector_lng': pickup.collector_lng,
        'address_lat': address.latitude or 28.6139,
        'address_lng': address.longitude or 77.2090,
        'address': address.address_line
    })

@app.route('/user/rewards')
@login_required
@role_required('user')
def rewards():
    user = User.query.get(session['user_id'])
    transactions = RewardTransaction.query.filter_by(user_id=user.id).order_by(RewardTransaction.created_at.desc()).all()
    return render_template('rewards.html', user=user, transactions=transactions)

@app.route('/user/redeem_points', methods=['POST'])
@login_required
@role_required('user')
def redeem_points():
    user = User.query.get(session['user_id'])
    points_to_redeem = int(request.form['points'])
    
    if points_to_redeem <= 0:
        flash('Invalid points amount', 'danger')
        return redirect(url_for('rewards'))
    
    if user.points >= points_to_redeem:
        user.points -= points_to_redeem
        transaction = RewardTransaction(
            user_id=user.id,
            points=-points_to_redeem,
            action='redeemed',
            description=f'Redeemed {points_to_redeem} points for discount voucher'
        )
        db.session.add(transaction)
        db.session.commit()
        flash(f'Successfully redeemed {points_to_redeem} points!', 'success')
        create_notification(user.id, f'You redeemed {points_to_redeem} points. Enjoy your reward!')
    else:
        flash('Insufficient points', 'danger')
    
    return redirect(url_for('rewards'))

@app.route('/user/complaint', methods=['POST'])
@login_required
@role_required('user')
def submit_complaint():
    subject = request.form['subject']
    description = request.form['description']
    complaint = Complaint(user_id=session['user_id'], subject=subject, description=description)
    db.session.add(complaint)
    db.session.commit()
    flash('Complaint submitted successfully', 'success')
    return redirect(url_for('user_dashboard'))

# ------------------------- Collector Panel -------------------------
@app.route('/collector/dashboard')
@login_required
@role_required('collector')
def collector_panel():
    collector = User.query.get(session['user_id'])
    pending_pickups = PickupRequest.query.filter_by(status='pending').all()
    accepted_pickups = PickupRequest.query.filter_by(collector_id=collector.id, status='accepted').all()
    in_progress = PickupRequest.query.filter_by(collector_id=collector.id, status='in_progress').all()
    completed = PickupRequest.query.filter_by(collector_id=collector.id, status='completed').order_by(PickupRequest.completed_at.desc()).limit(10).all()
    
    return render_template('collector_panel.html', 
                         collector=collector, 
                         pending_pickups=pending_pickups,
                         accepted_pickups=accepted_pickups,
                         in_progress=in_progress,
                         completed=completed)

@app.route('/collector/accept/<int:pickup_id>')
@login_required
@role_required('collector')
def accept_pickup(pickup_id):
    pickup = PickupRequest.query.get_or_404(pickup_id)
    if pickup.status != 'pending':
        flash('Pickup already assigned', 'warning')
        return redirect(url_for('collector_panel'))
    
    pickup.collector_id = session['user_id']
    pickup.status = 'accepted'
    db.session.commit()
    
    create_notification(pickup.user_id, f'Your pickup request #{pickup.id} has been accepted by collector {User.query.get(session["user_id"]).name}')
    flash('Pickup accepted', 'success')
    return redirect(url_for('collector_panel'))

@app.route('/collector/update_status/<int:pickup_id>', methods=['POST'])
@login_required
@role_required('collector')
def update_pickup_status(pickup_id):
    pickup = PickupRequest.query.get_or_404(pickup_id)
    if pickup.collector_id != session['user_id']:
        abort(403)
    
    new_status = request.form['status']
    pickup.status = new_status
    
    if new_status == 'completed':
        pickup.completed_at = datetime.datetime.utcnow()
        # Award points to user
        points_earned = 10 + int(pickup.weight_estimate or 1) * 5
        pickup.points_earned = points_earned
        user = User.query.get(pickup.user_id)
        user.points += points_earned
        
        transaction = RewardTransaction(
            user_id=user.id,
            points=points_earned,
            action='earned',
            description=f'Points earned for pickup #{pickup.id} - {pickup.waste_type} waste'
        )
        db.session.add(transaction)
        create_notification(pickup.user_id, f'Pickup #{pickup.id} completed! You earned {points_earned} points!')
    
    db.session.commit()
    flash(f'Pickup status updated to {new_status}', 'success')
    return redirect(url_for('collector_panel'))

@app.route('/collector/update_location/<int:pickup_id>', methods=['POST'])
@login_required
@role_required('collector')
def update_collector_location(pickup_id):
    pickup = PickupRequest.query.get_or_404(pickup_id)
    if pickup.collector_id != session['user_id']:
        abort(403)
    
    data = request.get_json()
    pickup.collector_lat = data.get('lat')
    pickup.collector_lng = data.get('lng')
    db.session.commit()
    return jsonify({'success': True})

@app.route('/collector/route_view')
@login_required
@role_required('collector')
def route_view():
    collector = User.query.get(session['user_id'])
    assigned_pickups = PickupRequest.query.filter_by(collector_id=collector.id).filter(PickupRequest.status.in_(['accepted', 'in_progress'])).all()
    pickups_data = []
    for pickup in assigned_pickups:
        address = pickup.address
        pickups_data.append({
            'id': pickup.id,
            'waste_type': pickup.waste_type,
            'status': pickup.status,
            'scheduled_date': pickup.scheduled_date.isoformat() if pickup.scheduled_date else '',
            'address': {
                'address_line': address.address_line if address else '',
                'area': address.area if address else '',
                'latitude': address.latitude if address and address.latitude is not None else None,
                'longitude': address.longitude if address and address.longitude is not None else None
            }
        })
    return render_template('route_view.html', pickups=assigned_pickups, pickups_data=pickups_data)

# ------------------------- Admin Panel -------------------------
@app.route('/admin/dashboard')
@login_required
@role_required('admin')
def admin_dashboard():
    total_users = User.query.filter_by(role='user').count()
    total_collectors = User.query.filter_by(role='collector').count()
    total_pickups = PickupRequest.query.count()
    completed_pickups = PickupRequest.query.filter_by(status='completed').count()
    total_points_awarded = db.session.query(func.sum(RewardTransaction.points)).filter(RewardTransaction.action=='earned').scalar() or 0
    pending_complaints = Complaint.query.filter_by(status='pending').count()
    
    # Monthly trends data
    monthly_data = db.session.query(
        extract('month', PickupRequest.created_at).label('month'),
        func.count(PickupRequest.id).label('count')
    ).filter(extract('year', PickupRequest.created_at) == 2025).group_by('month').all()
    
    months = [f"Month {int(m[0])}" for m in monthly_data]
    counts = [int(m[1]) for m in monthly_data]
    
    return render_template('admin_dashboard.html', 
                         total_users=total_users,
                         total_collectors=total_collectors,
                         total_pickups=total_pickups,
                         completed_pickups=completed_pickups,
                         total_points_awarded=total_points_awarded,
                         pending_complaints=pending_complaints,
                         monthly_labels=months,
                         monthly_data=counts)

@app.route('/admin/users')
@login_required
@role_required('admin')
def admin_users():
    users = User.query.all()
    return render_template('admin_users.html', users=users)

@app.route('/admin/collectors')
@login_required
@role_required('admin')
def admin_collectors():
    collectors = User.query.filter_by(role='collector').all()
    return render_template('admin_collectors.html', collectors=collectors)

@app.route('/admin/complaints')
@login_required
@role_required('admin')
def admin_complaints():
    complaints = Complaint.query.order_by(Complaint.created_at.desc()).all()
    return render_template('admin-complaints.html', complaints=complaints)

@app.route('/admin/complaint/resolve/<int:complaint_id>')
@login_required
@role_required('admin')
def resolve_complaint(complaint_id):
    complaint = Complaint.query.get_or_404(complaint_id)
    complaint.status = 'resolved'
    db.session.commit()
    flash('Complaint resolved', 'success')
    return redirect(url_for('admin_complaints'))

@app.route('/admin/reports')
@login_required
@role_required('admin')
def admin_reports():
    # Waste type distribution
    waste_stats = db.session.query(PickupRequest.waste_type, func.count(PickupRequest.id)).group_by(PickupRequest.waste_type).all()
    waste_types = [w[0] for w in waste_stats]
    waste_counts = [w[1] for w in waste_stats]
    
    # Area-wise stats
    area_stats = db.session.query(Address.area, func.count(PickupRequest.id)).join(PickupRequest).group_by(Address.area).all()
    areas = [a[0] for a in area_stats]
    area_counts = [a[1] for a in area_stats]
    
    # Monthly trends
    monthly_data = db.session.query(
        func.strftime('%Y-%m', PickupRequest.created_at).label('month'),
        func.count(PickupRequest.id).label('count')
    ).group_by('month').order_by('month').limit(12).all()
    
    months = [m[0] for m in monthly_data]
    monthly_counts = [m[1] for m in monthly_data]
    
    return render_template('admin_reports.html', 
                         waste_types=waste_types, 
                         waste_counts=waste_counts,
                         areas=areas,
                         area_counts=area_counts,
                         months=months,
                         monthly_counts=monthly_counts)

@app.route('/api/notifications/unread')
@login_required
def unread_notifications():
    count = Notification.query.filter_by(user_id=session['user_id'], is_read=False).count()
    notifications = Notification.query.filter_by(user_id=session['user_id'], is_read=False).order_by(Notification.created_at.desc()).limit(10).all()
    return jsonify({'count': count, 'notifications': [{'id': n.id, 'message': n.message} for n in notifications]})

@app.route('/api/notifications/mark_read/<int:notif_id>')
@login_required
def mark_notification_read(notif_id):
    notif = Notification.query.get_or_404(notif_id)
    if notif.user_id == session['user_id']:
        notif.is_read = True
        db.session.commit()
    return jsonify({'success': True})

# ------------------------- Create Database -------------------------
with app.app_context():
    db.create_all()

    admin_email = os.environ.get('DEFAULT_ADMIN_EMAIL')
    admin_password = os.environ.get('DEFAULT_ADMIN_PASSWORD')
    admin_phone = os.environ.get('DEFAULT_ADMIN_PHONE', '9999999999')
    admin = User.query.filter_by(role='admin').first()
    if admin_email and admin_password:
        if admin:
            admin.email = admin_email
            admin.password = hashlib.sha256(admin_password.encode()).hexdigest()
            admin.phone = admin_phone
        else:
            admin = User(
                name='Admin',
                email=admin_email,
                password=hashlib.sha256(admin_password.encode()).hexdigest(),
                phone=admin_phone,
                role='admin'
            )
            db.session.add(admin)
        db.session.commit()
    elif admin and admin.password == '240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9':
        admin.password = hashlib.sha256(uuid.uuid4().hex.encode()).hexdigest()
        db.session.commit()

    collector_email = os.environ.get('DEFAULT_COLLECTOR_EMAIL')
    collector_password = os.environ.get('DEFAULT_COLLECTOR_PASSWORD')
    collector_phone = os.environ.get('DEFAULT_COLLECTOR_PHONE', '8888888888')
    collector = User.query.filter_by(role='collector').first()
    if collector_email and collector_password:
        if collector:
            collector.email = collector_email
            collector.password = hashlib.sha256(collector_password.encode()).hexdigest()
            collector.phone = collector_phone
        else:
            collector = User(
                name='Collector',
                email=collector_email,
                password=hashlib.sha256(collector_password.encode()).hexdigest(),
                phone=collector_phone,
                role='collector'
            )
            db.session.add(collector)
        db.session.commit()
    elif collector and collector.password == 'f792a67c1e3d1bd14b5324c1649611047129dbc71927e2767f4413e87e3791c7':
        collector.password = hashlib.sha256(uuid.uuid4().hex.encode()).hexdigest()
        db.session.commit()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(debug=debug, host='0.0.0.0', port=port)
