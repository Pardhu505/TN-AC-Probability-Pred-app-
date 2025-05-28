document.addEventListener('DOMContentLoaded', () => {
  const form = document.querySelector('form');

  if (form) {
    form.addEventListener('submit', (event) => {
      event.preventDefault(); // Prevent default form submission

      const usernameInput = document.getElementById('username');
      const passwordInput = document.getElementById('password');

      const username = usernameInput.value.trim();
      const password = passwordInput.value.trim();

      if (username === '' || password === '') {
        alert('Please enter both username and password.');
      } else {
        console.log('Form validation successful. Username:', username, 'Password:', password);
        // Here you would typically send the data to a server
        // For this example, we're just logging to the console.
      }
    });
  } else {
    console.error('Login form not found!');
  }
});
