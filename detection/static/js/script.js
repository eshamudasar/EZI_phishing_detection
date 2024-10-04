document.addEventListener("DOMContentLoaded", function () {
    // Function to open the login form
    function openForm() {
        document.getElementById("myForm").style.display = "block";
    }

    // Function to close the login form
    function closeForm() {
        document.getElementById("myForm").style.display = "none";
    }

    // Example: Validate input before form submission
    const keywordForm = document.querySelector('#keyword-form');
    const urlForm = document.querySelector('#url-form');

    // Keyword form validation
    if (keywordForm) {
        keywordForm.addEventListener('submit', function (e) {
            const textInput = document.querySelector('#text');
            if (textInput.value.trim() === '') {
                e.preventDefault();
                alert('Please enter a message to check for phishing.');
            }
        });
    }

    // URL form validation
    if (urlForm) {
        urlForm.addEventListener('submit', function (e) {
            const urlInput = document.querySelector('#url');
            const urlPattern = /^(http|https):\/\/[^ "]+$/; // Basic URL validation
            if (!urlPattern.test(urlInput.value.trim())) {
                e.preventDefault();
                alert('Please enter a valid URL (e.g., https://example.com).');
            }
        });
    }
    
    // Assign open and close functions to buttons (if needed)
    document.querySelector('.open-btn').addEventListener('click', openForm);
    document.querySelector('.close-btn').addEventListener('click', closeForm);
});
