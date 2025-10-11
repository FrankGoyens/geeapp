// Game state
let currentNumber = 5;  // Starting with 5 as per the example
let gameStartTime;

// Initialize the game when the page loads
document.addEventListener('DOMContentLoaded', () => {
    initializeGame(currentNumber);
});

function initializeGame(number) {
    const apartmentBuilding = document.getElementById('apartment-building');
    gameStartTime = new Date();
    
    // Create building structure
    const buildingHtml = `
        <div class="roof">
            <input type="number" id="roof-number" value="${number}" min="1" max="100" aria-label="Target sum number">
        </div>
        <div class="game-instructions">
            Fill in each floor's right side so the numbers add up to ${number}!
        </div>
        <table class="floors">
            <thead>
                <tr>
                    <th>Left</th>
                    <th>Right</th>
                </tr>
            </thead>
            <tbody>
                ${generateFloors(number)}
            </tbody>
        </table>
        <div class="feedback-message"></div>
    `;
    
    apartmentBuilding.innerHTML = buildingHtml;
    setupInputListeners();
    setupRoofNumberListener();
}

function generateFloors(number) {
    // Create array with all numbers from 0 to number
    const numbers = Array.from({ length: number + 1 }, (_, i) => i);
    
    // Shuffle the array using Fisher-Yates algorithm
    for (let i = numbers.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [numbers[i], numbers[j]] = [numbers[j], numbers[i]];
    }
    
    // Generate HTML for each floor
    let floorsHtml = '';
    numbers.forEach(leftNumber => {
        floorsHtml += `
            <tr class="floor">
                <td>${leftNumber}</td>
                <td>
                    <input type="number" min="0" max="${number}" 
                           class="answer-input" data-left="${leftNumber}"
                           aria-label="Enter a number that adds up to ${number}">
                </td>
            </tr>
        `;
    });
    return floorsHtml;
}

function setupInputListeners() {
    const inputs = document.querySelectorAll('.answer-input');
    const feedbackMessage = document.querySelector('.feedback-message');
    
    inputs.forEach(input => {
        input.addEventListener('input', (e) => {
            const value = e.target.value;
            const leftNumber = parseInt(e.target.dataset.left);
            
            // Clear previous feedback
            feedbackMessage.textContent = '';
            
            // Validate input
            if (value === '') {
                e.target.classList.remove('correct', 'incorrect');
                return;
            }
            
            const numValue = parseInt(value);
            
            // Check if input is valid
            if (isNaN(numValue) || numValue < 0 || numValue > currentNumber) {
                e.target.classList.remove('correct');
                e.target.classList.add('incorrect');
                feedbackMessage.textContent = `Please enter a number between 0 and ${currentNumber}`;
                return;
            }
            
            // Check if answer is correct
            if (numValue + leftNumber === currentNumber) {
                e.target.classList.remove('incorrect');
                e.target.classList.add('correct');
                playSuccessSound();
            } else {
                e.target.classList.remove('correct');
                e.target.classList.add('incorrect');
                
                // Give a hint
                if (numValue + leftNumber > currentNumber) {
                    feedbackMessage.textContent = "That's too big! Try a smaller number.";
                } else {
                    feedbackMessage.textContent = "That's too small! Try a bigger number.";
                }
            }
            
            checkAllAnswers();
        });
    });
}

function playSuccessSound() {
    // You can add a success sound here if desired
    // For now, we'll just add a visual feedback
    const input = document.activeElement;
    if (input) {
        input.style.animation = 'correct-answer 0.5s';
        setTimeout(() => {
            input.style.animation = '';
        }, 500);
    }
}

function setupRoofNumberListener() {
    const roofInput = document.getElementById('roof-number');
    const feedbackMessage = document.querySelector('.feedback-message');

    roofInput.addEventListener('input', (e) => {
        const value = parseInt(e.target.value);
        
        // Clear any previous feedback
        feedbackMessage.textContent = '';
        
        // Validate input
        if (isNaN(value) || value < 1 || value > 100) {
            feedbackMessage.textContent = 'Please enter a number between 1 and 100';
            roofInput.classList.add('invalid');
            return;
        }
        
        roofInput.classList.remove('invalid');
        currentNumber = value;
        initializeGame(value);
    });
}

function checkAllAnswers() {
    const inputs = document.querySelectorAll('.answer-input');
    const allCorrect = Array.from(inputs).every(input => 
        input.classList.contains('correct')
    );
    
    if (allCorrect) {
        const timeTaken = Math.round((new Date() - gameStartTime) / 1000);
        const feedbackMessage = document.querySelector('.feedback-message');
        
        // Show completion message with time
        feedbackMessage.innerHTML = `
            <div class="success-message">
                <h3>🎉 Congratulations! 🎉</h3>
                <p>You completed the apartment building in ${timeTaken} seconds!</p>
                <button onclick="initializeGame(currentNumber)">Play Again</button>
            </div>
        `;
        
        // Add celebration animation to the building
        document.getElementById('apartment-building').classList.add('celebration');
    }
}