#!/usr/bin/env python3
"""
Test script for Agent Chat functionality.

This script tests the chat feature by simulating conversations with each agent.
It verifies that agents can respond contextually to questions about their analysis.

Usage:
    python test_agent_chat.py
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.agents.genotype_agent import GenotypeAgent
from src.agents.phenotype_agent import PhenotypeAgent
from src.agents.environment_agent import EnvironmentAgent
from src.data_processing.data_loader import DataLoader
from src.config.settings import config

def test_agent_chat():
    """Test chat functionality for all agents"""
    
    print("=" * 80)
    print("Testing Agent Chat Functionality")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading data...")
    data_loader = DataLoader()
    try:
        raw_data = data_loader.load_data()
        processed_data = data_loader.preprocess_data()
        print(f"   ✓ Data loaded successfully: {len(raw_data)} entries")
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        return False
    
    # Initialize agents
    print("\n2. Initializing agents...")
    genotype_agent = GenotypeAgent()
    phenotype_agent = PhenotypeAgent()
    environment_agent = EnvironmentAgent()
    
    # Set data for agents
    genotype_agent.set_data(processed_data)
    phenotype_agent.set_data(processed_data)
    environment_agent.set_data(processed_data)
    print("   ✓ Agents initialized")
    
    # Run quick analysis to generate context
    print("\n3. Running quick analyses to generate context...")
    try:
        genotype_analysis = genotype_agent.analyze(
            "Analyze genetic diversity",
            {"analysis_type": "diversity"}
        )
        phenotype_analysis = phenotype_agent.analyze(
            "Analyze trait correlations",
            {"analysis_type": "trait_correlation"}
        )
        environment_analysis = environment_agent.analyze(
            "Analyze location effects",
            {"analysis_type": "location_effects"}
        )
        print("   ✓ Analyses completed")
        print(f"     - Genotype: {genotype_analysis.get('status', 'unknown')}")
        print(f"     - Phenotype: {phenotype_analysis.get('status', 'unknown')}")
        print(f"     - Environment: {environment_analysis.get('status', 'unknown')}")
    except Exception as e:
        print(f"   ✗ Error running analyses: {e}")
        return False
    
    # Test chat with each agent
    print("\n4. Testing chat functionality...")
    
    # Test Genotype Agent
    print("\n   --- Testing Genotype Agent ---")
    test_questions_genotype = [
        "What did you find in your genetic diversity analysis?",
        "Which lines have the highest diversity scores?"
    ]
    
    for i, question in enumerate(test_questions_genotype, 1):
        print(f"\n   Question {i}: {question}")
        try:
            response = genotype_agent.chat(
                user_message=question,
                chat_history=[],
                analysis_context={"diversity": genotype_analysis}
            )
            print(f"   Response: {response[:200]}..." if len(response) > 200 else f"   Response: {response}")
            print("   ✓ Genotype agent chat working")
        except Exception as e:
            print(f"   ✗ Error: {e}")
            return False
    
    # Test Phenotype Agent
    print("\n   --- Testing Phenotype Agent ---")
    test_questions_phenotype = [
        "What trait correlations did you find?",
        "Which traits are most important for breeding decisions?"
    ]
    
    for i, question in enumerate(test_questions_phenotype, 1):
        print(f"\n   Question {i}: {question}")
        try:
            response = phenotype_agent.chat(
                user_message=question,
                chat_history=[],
                analysis_context={"correlations": phenotype_analysis}
            )
            print(f"   Response: {response[:200]}..." if len(response) > 200 else f"   Response: {response}")
            print("   ✓ Phenotype agent chat working")
        except Exception as e:
            print(f"   ✗ Error: {e}")
            return False
    
    # Test Environment Agent
    print("\n   --- Testing Environment Agent ---")
    test_questions_environment = [
        "What did you find about location effects?",
        "Which locations are most suitable for testing?"
    ]
    
    for i, question in enumerate(test_questions_environment, 1):
        print(f"\n   Question {i}: {question}")
        try:
            response = environment_agent.chat(
                user_message=question,
                chat_history=[],
                analysis_context={"location_effects": environment_analysis}
            )
            print(f"   Response: {response[:200]}..." if len(response) > 200 else f"   Response: {response}")
            print("   ✓ Environment agent chat working")
        except Exception as e:
            print(f"   ✗ Error: {e}")
            return False
    
    # Test conversation context
    print("\n5. Testing conversation context...")
    chat_history = [
        {"role": "user", "content": "What did you find in your analysis?"},
        {"role": "assistant", "content": "I analyzed genetic diversity across all lines..."}
    ]
    
    try:
        response = genotype_agent.chat(
            user_message="Can you elaborate on that?",
            chat_history=chat_history,
            analysis_context={"diversity": genotype_analysis}
        )
        print(f"   Response: {response[:200]}..." if len(response) > 200 else f"   Response: {response}")
        print("   ✓ Conversation context working")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("✓ All tests passed! Agent chat functionality is working correctly.")
    print("=" * 80)
    
    return True

def test_chat_without_analysis():
    """Test chat functionality when no analysis has been run"""
    
    print("\n" + "=" * 80)
    print("Testing Chat Without Prior Analysis")
    print("=" * 80)
    
    agent = GenotypeAgent()
    
    try:
        response = agent.chat(
            user_message="What can you tell me about genetic diversity?",
            chat_history=[],
            analysis_context=None
        )
        print(f"\nResponse: {response[:300]}..." if len(response) > 300 else f"\nResponse: {response}")
        print("\n✓ Agent can respond even without prior analysis context")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("\n🌱 AgriAgent - Agent Chat Feature Test Suite\n")
    
    # Check if API key is set
    if not config.ANTHROPIC_API_KEY or config.ANTHROPIC_API_KEY == "your-api-key-here":
        print("❌ ANTHROPIC_API_KEY not set in .env file")
        print("Please set your API key before running tests.")
        sys.exit(1)
    
    # Run tests
    success = True
    
    try:
        success = test_agent_chat()
        if success:
            success = test_chat_without_analysis()
    except KeyboardInterrupt:
        print("\n\n⚠️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    if success:
        print("\n✅ All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        sys.exit(1)

